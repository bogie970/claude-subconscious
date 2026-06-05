/**
 * Local file-based pattern store.
 * Loads/saves the 5 letta-style pattern blocks.
 * (The one-shot whisper queue was retired 2026-06-04.)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { fileURLToPath } from 'url';
import { Agent, MemoryBlock } from './conversation_utils.ts';

const AGENT_NAME = 'Hermes';

function getBlocksFilePath(cwd: string): string {
  const home = process.env.LETTA_HOME
    ? expandHome(process.env.LETTA_HOME)
    : cwd;
  return path.join(home, '.letta', 'claude', 'local_blocks.json');
}

function getSeedBlocksPath(): string {
  const thisDir = path.dirname(fileURLToPath(import.meta.url));
  return path.join(thisDir, '..', 'data', 'local_blocks.json');
}

function expandHome(p: string): string {
  const home = process.env.HOME || process.env.USERPROFILE || '';
  if (p === '~' || p === '$HOME' || p === '${HOME}') return home;
  if (p.startsWith('~/')) return path.join(home, p.slice(2));
  if (p.startsWith('$HOME/')) return path.join(home, p.slice(6));
  if (p.startsWith('${HOME}/')) return path.join(home, p.slice(8));
  return p;
}

interface BlocksFile {
  version: number;
  blocks: Record<string, {
    label: string;
    description: string;
    value: string;
    char_limit: number;
    updated_at: string;
  }>;
}

/**
 * Load memory blocks from local JSON file.
 * If the per-project file doesn't exist, copies from seed template.
 */
export function loadLocalBlocks(cwd: string): MemoryBlock[] {
  const blocksPath = getBlocksFilePath(cwd);

  if (!fs.existsSync(blocksPath)) {
    const seedPath = getSeedBlocksPath();
    if (fs.existsSync(seedPath)) {
      const dir = path.dirname(blocksPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.copyFileSync(seedPath, blocksPath);
    } else {
      return [];
    }
  }

  try {
    const raw: BlocksFile = JSON.parse(fs.readFileSync(blocksPath, 'utf-8'));
    return Object.values(raw.blocks)
      .filter(b => b.value && b.value.trim().length > 0)
      .map(b => ({
        label: b.label,
        description: b.description,
        value: b.value,
      }));
  } catch {
    return [];
  }
}

/**
 * Returns a fake Agent object backed by local blocks.
 * Drop-in replacement for fetchAgent() in local mode.
 */
export function getLocalAgent(cwd: string): Agent {
  return {
    id: 'local-agent',
    name: AGENT_NAME,
    description: 'Local subconscious (no cloud)',
    blocks: loadLocalBlocks(cwd),
  };
}

/**
 * Deterministic local conversation ID from session ID.
 */
export function getLocalConversationId(sessionId: string): string {
  const hash = crypto.createHash('sha256').update(`local-${sessionId}`).digest('hex').slice(0, 12);
  return `local-conv-${hash}`;
}

// The subconscious "whisper" feature (consumeWhispers + whispers.json queue)
// was retired 2026-06-04. The steward now occupies a real context-window slot.
// Do NOT re-introduce a whisper consumer here.
