// index.js — Yura Vision (ESM, robust Baileys import + grup-reply only + WA formatting strict)
import 'dotenv/config'
import pino from 'pino'
import NodeCache from 'node-cache'
import qrcode from 'qrcode-terminal'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import Tesseract from 'tesseract.js'
import { GoogleGenerativeAI } from '@google/generative-ai'

// ====== Konfigurasi & Logger ======
const logger = pino({
  level: process.env.LOG_LEVEL || 'warn', // set "info" sementara untuk debug
  base: undefined,
  timestamp: pino.stdTimeFunctions.isoTime
})

const GENAI_KEY   = process.env.GEMINI_API_KEY
const GENAI_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash' // Menggunakan model flash yang lebih baru
const OCR_LANG    = process.env.OCR_LANG || 'eng+ind'
const AUTH_DIR    = process.env.AUTH_DIR || './auth'
const OWNER_LOCAL = (process.env.OWNER_LOCAL || '').replace(/\D/g, '') // contoh: 62821xxxxxxx

if (!GENAI_KEY) {
  logger.error('GEMINI_API_KEY belum di-set di .env')
  process.exit(1)
}
const genAI = new GoogleGenerativeAI(GENAI_KEY)

const __filename = fileURLToPath(import.meta.url)
const __dirname  = path.dirname(__filename)

const msgRetryCounterCache = new NodeCache()

// ====== Dynamic import Baileys (robust ke versi berbeda) ======
const Baileys = await import('@whiskeysockets/baileys')
const makeWASocket =
  Baileys.default || Baileys.makeWASocket
const useMultiFileAuthState =
  Baileys.useMultiFileAuthState || Baileys.default?.useMultiFileAuthState
const useSingleFileAuthState =
  Baileys.useSingleFileAuthState || Baileys.default?.useSingleFileAuthState
const DisconnectReason =
  Baileys.DisconnectReason || Baileys.default?.DisconnectReason
const downloadMediaMessage =
  Baileys.downloadMediaMessage || Baileys.default?.downloadMediaMessage

if (typeof makeWASocket !== 'function') {
  logger.error('makeWASocket tidak ditemukan dari @whiskeysockets/baileys. Pastikan versinya terbaru.')
  process.exit(1)
}
if (typeof useMultiFileAuthState !== 'function' && typeof useSingleFileAuthState !== 'function') {
  logger.error('Baileys tidak mengekspor useMultiFileAuthState maupun useSingleFileAuthState. Update ke versi terbaru.')
  process.exit(1)
}

// ====== State Memori ======
const chatStates = new Map() // per chatId: { history: [{role,content}], ocrDocs: [{id,text,ts}], chatId }
const MAX_HISTORY_ITEMS = 20
const MAX_OCR_DOCS = 5

function ensureChatState(chatId) {
  if (!chatStates.has(chatId)) chatStates.set(chatId, { chatId, history: [], ocrDocs: [] })
  return chatStates.get(chatId)
}
function pushHistory(state, role, content) {
  state.history.push({ role, content })
  if (state.history.length > MAX_HISTORY_ITEMS) {
    state.history.splice(0, state.history.length - MAX_HISTORY_ITEMS)
  }
}
function addOcrDoc(state, text) {
  const id = `OCR-${Date.now()}`
  state.ocrDocs.push({ id, text, ts: Date.now() })
  if (state.ocrDocs.length > MAX_OCR_DOCS) {
    state.ocrDocs.splice(0, state.ocrDocs.length - MAX_OCR_DOCS)
  }
  return id
}
function summarize(text, limit = 800) {
  const t = (text || '').replace(/\s+/g, ' ').trim()
  return t.length > limit ? t.slice(0, limit) + '…' : t
}

// ====== JID & Message Helpers ======
function jidLocal(jid) {
  if (!jid) return ''
  return String(jid).split('@')[0].split(':')[0] // "628xxx:2@s.whatsapp.net" -> "628xxx"
}
function unwrapMessage(msg) {
  let inner = msg
  if (inner?.ephemeralMessage) inner = inner.ephemeralMessage.message
  if (inner?.viewOnceMessageV2) inner = inner.viewOnceMessageV2.message
  if (inner?.viewOnceMessage)   inner = inner.viewOnceMessage.message
  if (inner?.editedMessage)     inner = inner.editedMessage.message
  if (inner?.documentWithCaptionMessage) inner = inner.documentWithCaptionMessage.message
  return inner || msg
}
// Fungsi getContextInfo sudah tidak diperlukan lagi dan bisa dihapus.

function resetSenderKeysForGroup(groupJid) {
  try {
    const files = fs.readdirSync(AUTH_DIR).filter(f => f.startsWith(`sender-key-${groupJid}`))
    for (const f of files) fs.unlinkSync(path.join(AUTH_DIR, f))
    return files.length
  } catch {
    return 0
  }
}

// ====== Unicode Bold (judul tanpa bintang) ======
const BOLD_MAP = (() => {
  const map = {}
  const a = 'abcdefghijklmnopqrstuvwxyz', A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', d = '0123456789'
  const boldA = [...'𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𐐰']
  const bolda = [...'𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳']
  const boldd = [...'𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗']
  for (let i=0;i<26;i++) { map[A[i]] = boldA[i]; map[a[i]] = bolda[i] }
  for (let i=0;i<10;i++) { map[d[i]] = boldd[i] }
  return map
})()
function toUnicodeBold(s='') {
  return s.split('').map(ch => BOLD_MAP[ch] || ch).join('')
}

// ====== Formatter WhatsApp ketat (tanpa **double**) ======
function formatForWhatsAppStrict(text) {
  if (!text) return ''
  const parts = text.split(/```/g) // Lindungi blok kode
  for (let i = 0; i < parts.length; i += 2) {
    let seg = parts[i]
    // Heading: baris penuh **...** / __...__ / #/##/###
    seg = seg.split('\n').map(line => {
      let m = line.match(/^\s*\*\*(.+?)\*\*\s*$/) || line.match(/^\s*__(.+?)__\s*$/)
      if (!m) {
        const mh = line.match(/^\s*#{1,6}\s+(.+?)\s*$/)
        if (mh) return toUnicodeBold(mh[1].trim())
      } else {
        return toUnicodeBold(m[1].trim())
      }
      return line
    }).join('\n')
    // Inline styles non-heading
    seg = seg.replace(/\*\*(.+?)\*\*/g, '*$1*')  // **bold** -> *bold*
             .replace(/__(.+?)__/g, '_$1_')      // __it__   -> _it_
             .replace(/~~(.+?)~~/g, '~$1~')      // ~~x~~    -> ~x~
             .replace(/[ \t]+$/gm, '')           // trim kanan
    parts[i] = seg
  }
  return parts.join('```')
}

// ====== Persona ======
const SYSTEM_PERSONA = `
Kamu adalah "Yura Naomi", adik ceria maskot Gachaverse.
Gaya: hangat, semangat, helpful, sopan, satu kaomoji ringan saat pas (mis. (˶ᵔ ᵕ ᵔ˶)).
Aturan:
- Gunakan format WhatsApp: *tebal*, _miring_, ~coret~, blok kode dengan tiga backtick.
- Hindari **double asterisk**. Untuk judul, gunakan huruf tebal Unicode (tanpa bintang).
- Jawab ringkas, jelas; boleh poin/nomor bila cocok.
- Jika merujuk teks OCR, sebut "dari gambar sebelumnya" atau pakai ID OCR.
- Jangan mengarang; bila ragu, minta klarifikasi singkat.
- Variasikan pembuka/penutup agar tidak monoton.
`.trim()

function buildPrompt(state, userTurn) {
  const kb = state.ocrDocs.length
    ? `\n\n[Pengetahuan dari OCR Terakhir]\n` +
      state.ocrDocs.slice(-3).map(d => `- ${d.id} (${new Date(d.ts).toLocaleString()}): ${summarize(d.text, 600)}`).join('\n')
    : ''
  const convo = state.history.map(h => `${h.role === 'user' ? 'User' : 'Yura'}: ${h.content}`).join('\n')
  const turn  = userTurn ? `\nUser: ${userTurn}` : ''
  return `${SYSTEM_PERSONA}

[Riwayat Percakapan]
${convo || '(Belum ada riwayat)'}
${kb}

Instruksi:
- Jawab sebagai "Yura Naomi".
- Jika pertanyaan merujuk "tadi"/gambar/OCR, gunakan ringkasan di [Pengetahuan dari OCR Terakhir].
- Hindari output terlalu panjang.

${turn}
Yura:`.trim()
}

async function callGemini(prompt) {
  const model = genAI.getGenerativeModel({ model: GENAI_MODEL })
  const result = await model.generateContent(prompt)
  const resp = await result.response.text()
  return (resp || '').trim()
}

// ====== Bot ======
async function startBot() {
  // Auth: prefer multi-file, fallback single-file
  let authState, saveCreds
  if (typeof useMultiFileAuthState === 'function') {
    ({ state: authState, saveCreds } = await useMultiFileAuthState(AUTH_DIR))
  } else {
    ({ state: authState, saveCreds } = await useSingleFileAuthState(path.join(AUTH_DIR, 'auth.json')))
    logger.warn('useMultiFileAuthState tidak tersedia; fallback ke single-file auth.')
  }

  const sock = makeWASocket({
    auth: authState,
    printQRInTerminal: false,
    logger,
    msgRetryCounterMap: msgRetryCounterCache
  })

  // Koneksi
  sock.ev.on('connection.update', ({ connection, lastDisconnect, qr }) => {
    if (qr) { logger.warn('QR tersedia — scan via WhatsApp Web'); qrcode.generate(qr, { small: true }) }
    if (connection === 'open')  logger.info('✅ Terhubung ke WhatsApp')
    if (connection === 'close') logger.error({ code: lastDisconnect?.error?.output?.statusCode }, '❌ Koneksi tertutup')
  })

  // Pesan masuk
  sock.ev.on('messages.upsert', async ({ messages }) => {
    const m = messages?.[0]
    if (!m?.message || m.key.fromMe) return

    const chatId  = m.key.remoteJid
    const isGroup = chatId.endsWith('@g.us')

    // Unwrap & siapkan WAMessage penuh
    const rawMsg  = unwrapMessage(m.message)
    const fullMsg = { ...m, message: rawMsg }

    // Ambil teks/caption
    let userText = ''
    if (rawMsg?.conversation) userText = rawMsg.conversation.trim()
    else if (rawMsg?.extendedTextMessage?.text) userText = rawMsg.extendedTextMessage.text.trim()
    else if (rawMsg?.imageMessage?.caption) userText = rawMsg.imageMessage.caption.trim()

    // === Perintah maintenance: !fix (owner saja) ===
    const senderLocal = (m.key.participant ? m.key.participant : m.key.remoteJid)?.split('@')[0]?.split(':')[0] || ''
    if (isGroup && userText?.toLowerCase() === '!fix' && OWNER_LOCAL && senderLocal.endsWith(OWNER_LOCAL)) {
      const n = resetSenderKeysForGroup(chatId)
      await sock.sendMessage(chatId, {
        text: n
          ? `𝐒𝐞𝐬𝐢 𝐠𝐫𝐮𝐩 𝐝𝐢𝐫𝐞𝐬𝐞𝐭 (${n} berkas). Kirim 1 pesan teks biasa ya agar SenderKey baru terkirim.`
          : `Tidak ada sender-key yang dihapus. Coba kirim 1 pesan teks biasa untuk refresh SenderKey.`
      }, { quoted: m })
      return
    }

    // ==========================================================
    // == BLOK PERBAIKAN: Logika untuk merespons di grup ==
    // ==========================================================
    if (isGroup) {
      // LANGSUNG ambil contextInfo dari pesan yang sudah di-unwrap. Ini cara yang benar.
      const ctx = rawMsg.contextInfo;
      const myLocal = jidLocal(sock.user?.id);

      // 1. Cek apakah bot di-mention.
      // Gunakan optional chaining (?.) dan nullish coalescing (|| []) agar aman jika ctx atau mentionedJid tidak ada.
      const isMentioned = (ctx?.mentionedJid || []).some(j => jidLocal(j) === myLocal);

      // 2. Cek apakah pesan ini adalah balasan (reply) untuk pesan bot.
      // `ctx.participant` berisi JID pengirim pesan yang di-reply.
      const isReplyToBot = jidLocal(ctx?.participant) === myLocal;

      // Jika tidak di-mention DAN tidak me-reply bot, abaikan pesan ini.
      if (!isMentioned && !isReplyToBot) {
        return;
      }
      
      // (Opsional tapi penting) Hapus mention dari teks agar tidak masuk ke prompt AI
      if (isMentioned) {
          userText = userText.replace(`@${myLocal}`, '').trim();
      }
    }
    // ==========================================================
    // == AKHIR BLOK PERBAIKAN ==
    // ==========================================================

    const state = ensureChatState(chatId)

    // Gambar → OCR
    if (rawMsg?.imageMessage) {
      try {
        const buffer = await downloadMediaMessage(fullMsg, 'buffer', {}, { logger })
        const dir = path.join(__dirname, 'download')
        fs.mkdirSync(dir, { recursive: true })
        const filePath = path.join(dir, `${Date.now()}.jpg`)
        fs.writeFileSync(filePath, buffer)

        const { data: { text } } = await Tesseract.recognize(filePath, OCR_LANG, { logger: () => {} })
        const ocrText = (text || '').trim()

        const minLen = 8
        if (!ocrText || ocrText.replace(/\s+/g, '').length < minLen) {
          await sock.sendMessage(chatId, { text: '⚠️ Gambarnya agak blur, OCR belum bisa baca. Coba foto lebih terang & tegak lurus ya.' }, { quoted: m })
          logger.warn('OCR too short/unclear; skipped AI turn')
          return
        }

        const id = addOcrDoc(state, ocrText)
        pushHistory(
          state,
          'user',
          userText
            ? `${userText}\n\n[Teks dari ${id}]:\n${summarize(ocrText)}`
            : `[Teks dari ${id}]:\n${summarize(ocrText)}`
        )
      } catch (e) {
        logger.warn({ err: e?.message }, 'OCR error')
        await sock.sendMessage(chatId, { text: '⚠️ Gagal membaca gambar. Coba kirim ulang ya.' }, { quoted: m })
        return
      }
    } else if (userText) {
      pushHistory(state, 'user', userText)
    } else {
      // Jika tidak ada gambar dan tidak ada teks (misal: stiker, video), jangan proses
      return
    }

    // Panggil AI
    try {
      // Kirim "typing" status untuk memberitahu user bot sedang memproses
      await sock.sendPresenceUpdate('composing', chatId)

      const prompt = buildPrompt(state, userText || '')
      const aiText = await callGemini(prompt)
      const waText = formatForWhatsAppStrict(aiText)
      if (!waText) {
        await sock.sendMessage(chatId, { text: '⚠️ Aku belum menerima balasan. Boleh ulangi pesannya ya?' }, { quoted: m })
        return
      }

      await sock.sendMessage(chatId, { text: waText }, { quoted: m })
      pushHistory(state, 'assistant', waText)

    } catch (err) {
      const msg = String(err?.message || err)
      if (/PreKey|No session record|No SenderKeyRecord|InvalidMessageException|Bad MAC/i.test(msg)) {
        logger.warn('Signal transient error (suppressed)')
      } else {
        logger.error({ err }, 'AI error')
      }
      await sock.sendMessage(chatId, { text: '❌ Aduh, ada sedikit kendala saat menghubungi AI. Coba lagi beberapa saat ya.' }, { quoted: m })
    } finally {
      // Hapus status "typing" setelah selesai
      await sock.sendPresenceUpdate('paused', chatId)
    }
  })

  sock.ev.on('creds.update', saveCreds)
}

startBot()