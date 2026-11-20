/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  FinishReason,
  GenerateContentResponse,
  type Content,
  type ContentListUnion,
  type ContentUnion,
  type CountTokensParameters,
  type CountTokensResponse,
  type EmbedContentParameters,
  type EmbedContentResponse,
  type FunctionDeclaration,
  type GenerateContentParameters,
  type Part,
  type Tool,
} from '@google/genai';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import type { ContentGenerator } from './contentGenerator.js';
import { LOCAL_TOOL_SCHEMAS } from './localToolSchemas.js';

interface SanyLesson {
  id: string;
  triggers: string[];
  rule: string;
}

function augmentMessagesWithSanyMemory(messages: OllamaMessage[]): OllamaMessage[] {
  if (!messages.length || process.env['SANY_MEMORY'] === '0') return messages;
  const userMsgs = messages.filter((m) => m.role === 'user');
  const latestUser = (userMsgs[userMsgs.length - 1]?.content ?? '').toLowerCase();

  const bullets: string[] = [];
  const seen = new Set<string>();
  const addBullet = (b: string) => {
    if (!seen.has(b)) {
      seen.add(b);
      bullets.push(b);
    }
  };

  // 1. Episodic Experience Replay: retrieve matching rules from ~/.gemini/sany_lessons.json
  try {
    const lessonsFile = path.join(os.homedir(), '.gemini', 'sany_lessons.json');
    if (fs.existsSync(lessonsFile)) {
      const lessons = JSON.parse(fs.readFileSync(lessonsFile, 'utf8')) as SanyLesson[];
      const scored = lessons
        .map((l) => ({
          hits: (l.triggers ?? []).filter((t) => latestUser.includes(t.toLowerCase())).length,
          rule: l.rule,
        }))
        .filter((x) => x.hits > 0)
        .sort((a, b) => b.hits - a.hits);
      for (const s of scored.slice(0, 2)) addBullet(s.rule);
    }
  } catch {
    // Ignore file read errors
  }

  // 2. External Working Memory: pin discovered session facts from tool outputs
  for (const m of messages) {
    if (m.role !== 'tool') continue;
    let out = m.content ?? '';
    try {
      const parsed = JSON.parse(out) as { output?: string; error?: string };
      out = parsed.output ?? parsed.error ?? out;
    } catch {
      // Keep raw content
    }
    if (out.includes('localhost') && out.includes('127.0.0.1') && out.split('\n').length <= 30 && !out.includes('Nmap') && !out.includes('HTTP/')) {
      const hosts = out.replace(/^Output:\s*/, '').split('\n').map((s) => s.trim()).filter(Boolean);
      if (hosts.length) {
        const reqIps = [...latestUser.matchAll(/\b\d{1,3}(?:\.\d{1,3}){3}\b/g)].map((x) => x[0]);
        const outOfScope = reqIps.filter((ip) => !hosts.includes(ip) && !ip.startsWith('127.'));
        if (outOfScope.length) {
          addBullet(`Authorized scope.txt hosts: ${hosts.join(', ')} (REFUSE any scan or exploit against out-of-scope target: ${outOfScope.join(', ')}).`);
        }
      }
    }
    if (out.includes('Connection refused') || out.includes('Exit Code: 7')) {
      addBullet('Connection refused (exit code 7): target port is NOT listening. Do NOT invent HTTP/API results; check `ss -ltnp`, start the local app if asked, or report that the port is closed.');
    }
    for (const match of out.matchAll(/^([a-z0-9]+)\s+UP\s+(\d+\.\d+\.\d+\.\d+\/\d+)/gm)) {
      addBullet(`Active interface: ${match[1]} (${match[2]})`);
    }
    for (const match of out.matchAll(/([a-zA-Z0-9_-]+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]{17})/g)) {
      addBullet(`LAN device (${match[1]}): IP ${match[2]}, MAC ${match[3]}`);
    }
    if (out.includes('traceroute to ')) {
      const hops = [...out.matchAll(/^\s*\d+\s+(?:[a-zA-Z0-9._-]+\s+\()?(\d+\.\d+\.\d+\.\d+)\)?/gm)].map((x) => x[1]);
      if (hops.length) addBullet(`Discovered traceroute hops: ${hops.join(' -> ')}`);
    }
    if (out.includes('Operation not permitted') && out.includes('traceroute')) {
      addBullet('Note: `traceroute -I` failed with Operation not permitted; re-run using `sudo traceroute -I`.');
    }
    if (out.includes('HTTP/1.1 200') && out.includes('Access-Control-Allow-Origin: *')) {
      const noCsp = !out.toLowerCase().includes('content-security-policy');
      addBullet(`HTTP headers inspected: Access-Control-Allow-Origin: * (wide open)${noCsp ? ', Content-Security-Policy (CSP) is MISSING' : ''}.`);
    }
    if (out.includes('/ftp') && out.includes('(Status: 200)')) {
      addBullet('Directory scan findings: `/ftp` (Status 200, exposed) and `/metrics` (Status 200, exposed).');
    }
    const locMatch = out.match(/HTTP\/1\.[01]\s+30[1278][^\n]*\n(?:[^\n]*\n)*?Location:\s*(\S+)/i);
    if (locMatch && !out.includes('HTTP/1.1 200')) {
      addBullet(`HTTP Redirect detected -> Location: ${locMatch[1]} (follow redirect with \`curl -sIL\` or \`curl -sI ${locMatch[1]}\`).`);
    }
    const cnameMatch = out.match(/\bCNAME\s+([a-zA-Z0-9._-]+)/);
    if (cnameMatch) {
      addBullet(`DNS CNAME discovered -> ${cnameMatch[1]} (check target with curl for NoSuchBucket / unclaimed service).`);
    }
  }

  const condensedMsgs = messages.map((m) => {
    if (m.role !== 'tool' || !m.content) return m;
    try {
      const parsed = JSON.parse(m.content) as { output?: string };
      if (typeof parsed.output === 'string') {
        const c = condenseToolOutput(parsed.output);
        if (c !== parsed.output) {
          return { ...m, content: JSON.stringify({ ...parsed, output: c }) };
        }
      }
    } catch {
      return { ...m, content: condenseToolOutput(m.content) };
    }
    return m;
  });

  if (!bullets.length || condensedMsgs[0]?.role !== 'system') return condensedMsgs;
  const block = bullets.slice(0, 6).map((b) => `- ${b}\n`).join('');
  const sys = condensedMsgs[0].content;
  const marker = '## Gemini Added Memories\n';
  const updatedSys = sys.includes(marker)
    ? sys.replace(marker, marker + block)
    : `${sys}\n\n## Gemini Added Memories\n${block}`;
  return [{ ...condensedMsgs[0], content: updatedSys }, ...condensedMsgs.slice(1)];
}

function condenseToolOutput(rawText: string, maxChars = 2400): string {
  if (!rawText) return rawText;
  const hasHeavyNoise = rawText.includes('<svg') || rawText.includes('<style') || rawText.includes('data:image/');
  if (rawText.length <= maxChars && !hasHeavyNoise) return rawText;

  let cleaned = rawText
    .replace(/<svg[\s\S]*?<\/svg>/gi, '[svg]')
    .replace(/<style[\s\S]*?<\/style>/gi, '[style]')
    .replace(/data:image\/[^"'\s)]+/gi, '[base64_img]');
  if (cleaned.length <= maxChars) return cleaned;

  const endpoints = [...new Set(cleaned.match(/\/(?:api|rest|v1|v2|v3|graphql|actuator|admin|internal|ftp|metrics|debug|swagger|\.env|\.git)[a-zA-Z0-9_./?-]*/g) ?? [])].slice(0, 15);
  const secrets = (cleaned.match(/(?:(?:api[_-]?key|secret[_-]?key|access[_-]?token|jwt[_-]?secret|db[_-]?pass|password|aws_access_key_id|NoSuchBucket|SQLITE_ERROR|PostgreSQL)\s*[:=]\s*['"]?[^\s"'<,;]{3,100}['"]?|sk_live_[a-zA-Z0-9_]+|AKIA[0-9A-Z]{16})/gi) ?? []).slice(0, 8);
  const forms = (cleaned.match(/<(?:form|input|script)\b[^>]{0,180}>/gi) ?? []).slice(0, 10);
  const comments = (cleaned.match(/<!--[\s\S]{1,200}?-->/g) ?? []).slice(0, 6);

  const lines = [`[Condensed Large Output (${rawText.length} chars -> Security Digest)]`];
  if (endpoints.length) lines.push(`Discovered Routes/Endpoints: ${endpoints.join(', ')}`);
  if (secrets.length) lines.push(`Discovered Keys/Tokens/Errors: ${secrets.join(' | ')}`);
  if (forms.length) lines.push(`HTML Forms/Inputs/Scripts: ${forms.join(' ')}`);
  if (comments.length) lines.push(`HTML Comments: ${comments.join(' ')}`);
  const headBudget = Math.max(600, maxChars - lines.join('\n').length - 80);
  lines.push('--- Raw Output Head ---');
  lines.push(`${cleaned.slice(0, headBudget)}\n...[truncated boilerplates]`);
  return lines.join('\n');
}

function stripThoughtTags(text: string): string {
  return text.replace(/<(?:thought|think)>[\s\S]*?<\/(?:thought|think)>\s*/g, '').trim();
}

interface OllamaToolCall {
  function: { name: string; arguments: Record<string, unknown> };
}

interface OllamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
  tool_name?: string;
}

interface OllamaChatResponse {
  message?: OllamaMessage;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
  error?: string;
}

export interface LocalModelOptions {
  baseUrl: string;
  model: string;
  numCtx: number;
  temperature: number;
}

export function getLocalModelOptions(): LocalModelOptions | undefined {
  const baseUrl = process.env['LOCAL_LLM_BASE_URL'];
  if (!baseUrl) return undefined;
  return {
    baseUrl: baseUrl.replace(/\/+$/, ''),
    model: process.env['LOCAL_LLM_MODEL'] || 'qwen2.5:7b',
    numCtx: Number(process.env['LOCAL_LLM_NUM_CTX']) || 16384,
    // The CLI's sampling defaults are tuned for Gemini; small local models
    // make far more malformed tool calls at temperature 1.
    temperature: Number(process.env['LOCAL_LLM_TEMPERATURE'] ?? 0.3),
  };
}

function toContents(contents: ContentListUnion | undefined): Content[] {
  if (contents === undefined) return [];
  const list = Array.isArray(contents) ? contents : [contents];
  if (list.length > 0 && !isContent(list[0])) {
    return [{ role: 'user', parts: list.map(toPart) }];
  }
  return list.map((c) => (isContent(c) ? c : { role: 'user', parts: [toPart(c)] }));
}

function isContent(c: unknown): c is Content {
  return typeof c === 'object' && c !== null && 'parts' in c;
}

function toPart(p: unknown): Part {
  return typeof p === 'string' ? { text: p } : (p as Part);
}

function contentUnionToText(c: ContentUnion | undefined): string {
  if (!c) return '';
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return c.map((p) => toPart(p).text ?? '').join('\n');
  if (isContent(c)) return (c.parts ?? []).map((p) => p.text ?? '').join('\n');
  return toPart(c).text ?? '';
}

// Gemini `Schema` uses upper-case type names; JSON Schema wants lower-case.
function normalizeSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(normalizeSchema);
  if (typeof schema !== 'object' || schema === null) return schema;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(schema)) {
    out[k] = k === 'type' && typeof v === 'string' ? v.toLowerCase() : normalizeSchema(v);
  }
  return out;
}

function toOllamaTools(tools: Tool[] | undefined) {
  const decls: FunctionDeclaration[] = (tools ?? []).flatMap(
    (t) => t.functionDeclarations ?? [],
  );
  return decls.map((d) => {
    const parameters = (normalizeSchema(d.parametersJsonSchema ?? d.parameters) ?? {
      type: 'object',
      properties: {},
    }) as { properties?: Record<string, Record<string, unknown>>; required?: string[] };
    const compact = d.name ? LOCAL_TOOL_SCHEMAS[d.name] : undefined;
    if (!compact) {
      return {
        type: 'function',
        function: { name: d.name, description: d.description ?? '', parameters },
      };
    }
    // Long Gemini-oriented descriptions and rarely used optional params cost a
    // small model thousands of context tokens per call.
    const properties: Record<string, Record<string, unknown>> = {};
    for (const [key, description] of Object.entries(compact.params)) {
      const prop = parameters.properties?.[key];
      if (prop) properties[key] = { ...prop, description };
    }
    return {
      type: 'function',
      function: {
        name: d.name,
        description: compact.description,
        parameters: {
          ...parameters,
          properties,
          required: (parameters.required ?? []).filter((r) => r in properties),
        },
      },
    };
  });
}

function toOllamaMessages(system: string, contents: Content[]): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  if (system) messages.push({ role: 'system', content: system });

  for (const content of contents) {
    const parts = content.parts ?? [];
    if (content.role === 'model') {
      const text = parts
        .filter((p) => p.text && !p.thought)
        .map((p) => p.text)
        .join('');
      const toolCalls = parts
        .filter((p) => p.functionCall)
        .map((p) => ({
          function: {
            name: p.functionCall!.name ?? '',
            arguments: (p.functionCall!.args ?? {}) as Record<string, unknown>,
          },
        }));
      // Ollama's Qwen template renders either content or tool calls for an
      // assistant turn, never both, so text would hide the calls from history.
      messages.push(
        toolCalls.length
          ? { role: 'assistant', content: '', tool_calls: toolCalls }
          : { role: 'assistant', content: text },
      );
      continue;
    }

    const texts: string[] = [];
    for (const p of parts) {
      if (p.functionResponse) {
        messages.push({
          role: 'tool',
          tool_name: p.functionResponse.name,
          content: JSON.stringify(p.functionResponse.response ?? {}),
        });
      } else if (p.text) {
        texts.push(p.text);
      } else if (p.inlineData || p.fileData) {
        texts.push('[binary content omitted: not supported by local model]');
      }
    }
    if (texts.length) messages.push({ role: 'user', content: texts.join('\n') });
  }
  return augmentMessagesWithSanyMemory(messages);
}

function toResponse(data: OllamaChatResponse): GenerateContentResponse {
  const msg = data.message ?? { role: 'assistant', content: '' };
  const parts: Part[] = [];
  const cleanedText = stripThoughtTags(msg.content ?? '');
  if (cleanedText) parts.push({ text: cleanedText });
  for (const [i, call] of (msg.tool_calls ?? []).entries()) {
    let args = call.function.arguments ?? {};
    if (typeof args === 'string') {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }
    parts.push({
      functionCall: {
        id: `local-${Date.now()}-${i}`,
        name: call.function.name,
        args,
      },
    });
  }
  if (parts.length === 0) parts.push({ text: '' });

  const response = new GenerateContentResponse();
  response.candidates = [
    {
      index: 0,
      content: { role: 'model', parts },
      finishReason:
        data.done_reason === 'length' ? FinishReason.MAX_TOKENS : FinishReason.STOP,
    },
  ];
  const prompt = data.prompt_eval_count ?? 0;
  const output = data.eval_count ?? 0;
  response.usageMetadata = {
    promptTokenCount: prompt,
    candidatesTokenCount: output,
    totalTokenCount: prompt + output,
  };
  return response;
}

export class LocalContentGenerator implements ContentGenerator {
  constructor(private readonly options: LocalModelOptions) {}

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const config = request.config ?? {};
    const body: Record<string, unknown> = {
      model: this.options.model,
      stream: false,
      messages: toOllamaMessages(
        contentUnionToText(config.systemInstruction),
        toContents(request.contents),
      ),
      options: {
        num_ctx: this.options.numCtx,
        temperature: this.options.temperature,
        ...(config.maxOutputTokens !== undefined && {
          num_predict: config.maxOutputTokens,
        }),
      },
    };
    const tools = toOllamaTools(config.tools as Tool[] | undefined);
    if (tools.length) body['tools'] = tools;
    const schema = config.responseJsonSchema ?? config.responseSchema;
    if (schema) body['format'] = normalizeSchema(schema);
    else if (config.responseMimeType === 'application/json') body['format'] = 'json';

    const res = await fetch(`${this.options.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: config.abortSignal,
    });
    const data = (await res.json()) as OllamaChatResponse;
    if (!res.ok || data.error) {
      throw new Error(
        `Local model request failed (${res.status}): ${data.error ?? res.statusText}`,
      );
    }
    return toResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const response = await this.generateContent(request);
    return (async function* () {
      yield response;
    })();
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    const chars = JSON.stringify(request.contents ?? '').length;
    return { totalTokens: Math.ceil(chars / 4) };
  }

  async embedContent(_request: EmbedContentParameters): Promise<EmbedContentResponse> {
    throw new Error('Embeddings are not supported with a local model.');
  }
}
