/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { fetchWithTimeout } from '../utils/fetch.js';

const SEARCH_TIMEOUT_MS = 15000;
const MAX_RESULTS = 8;
const USER_AGENT =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36';

export interface LocalSearchResult {
  title: string;
  url: string;
  snippet: string;
}

function stripTags(html: string): string {
  return html
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#0?39;|&#x27;/g, "'")
    .replace(/&nbsp;|&ensp;|&#183;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Bing wraps result links as /ck/a?...&u=a1<base64url of the real URL>.
function decodeBingUrl(href: string): string {
  const decoded = href.replace(/&amp;/g, '&');
  const match = /[?&]u=a1([^&]+)/.exec(decoded);
  if (!match) return decoded;
  try {
    return Buffer.from(match[1], 'base64url').toString('utf8');
  } catch {
    return decoded;
  }
}

async function searchSearxng(
  baseUrl: string,
  query: string,
  signal: AbortSignal,
): Promise<LocalSearchResult[]> {
  const url = `${baseUrl.replace(/\/+$/, '')}/search?format=json&q=${encodeURIComponent(query)}`;
  const res = await fetchWithTimeout(url, SEARCH_TIMEOUT_MS, { signal });
  if (!res.ok) throw new Error(`SearXNG returned ${res.status}`);
  const data = (await res.json()) as {
    results?: Array<{ title?: string; url?: string; content?: string }>;
  };
  return (data.results ?? []).slice(0, MAX_RESULTS).map((r) => ({
    title: r.title ?? '',
    url: r.url ?? '',
    snippet: r.content ?? '',
  }));
}

async function searchBing(
  query: string,
  signal: AbortSignal,
): Promise<LocalSearchResult[]> {
  const url = `https://www.bing.com/search?setlang=en&q=${encodeURIComponent(query)}`;
  const res = await fetchWithTimeout(url, SEARCH_TIMEOUT_MS, {
    signal,
    headers: { 'User-Agent': USER_AGENT, 'Accept-Language': 'en-US,en' },
  });
  if (!res.ok) throw new Error(`Bing returned ${res.status}`);
  const html = await res.text();
  const results: LocalSearchResult[] = [];
  for (const block of html.split('<li class="b_algo"').slice(1)) {
    const link = /<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/.exec(block);
    if (!link) continue;
    const snippet = /<p[^>]*>([\s\S]*?)<\/p>/.exec(block);
    results.push({
      title: stripTags(link[2]),
      url: decodeBingUrl(link[1]),
      snippet: snippet ? stripTags(snippet[1]) : '',
    });
    if (results.length >= MAX_RESULTS) break;
  }
  return results;
}

export async function localWebSearch(
  query: string,
  signal: AbortSignal,
): Promise<LocalSearchResult[]> {
  const searxng = process.env['SEARXNG_URL'];
  return searxng
    ? searchSearxng(searxng, query, signal)
    : searchBing(query, signal);
}

export function formatLocalSearchResults(
  query: string,
  results: LocalSearchResult[],
): string {
  if (results.length === 0) {
    return `No search results found for query: "${query}"`;
  }
  const lines = results.map(
    (r, i) => `[${i + 1}] ${r.title}\n    ${r.url}\n    ${r.snippet}`,
  );
  return `Web search results for "${query}":\n\n${lines.join('\n\n')}\n\nUse web_fetch on a result URL to read the full page.`;
}
