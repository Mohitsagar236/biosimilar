/* ═══════════════════════════════════════════
   RAG Assistant — app.js
   Features: Chat, Document management, Deepgram voice input
═══════════════════════════════════════════ */

/* ── State ── */
let selectedFiles   = [];
let isStreaming     = false;
let uploadPanelOpen = false;
let chats           = [];   // [{ id, title, messages: [{role,text,sources?}], createdAt }]
let activeChatId    = '';
const CHAT_STORAGE_KEY   = 'rag_chat_v1';
const CHATS_STORAGE_KEY  = 'rag_chats_v1';
const ACTIVE_CHAT_KEY    = 'rag_active_chat_v1';

function saveChatsToStorage() {
  try {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(chats));
    localStorage.setItem(ACTIVE_CHAT_KEY, activeChatId || '');
  } catch { /* quota/private */ }
}

function loadChatsFromStorage() {
  try {
    const raw = localStorage.getItem(CHATS_STORAGE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      if (Array.isArray(saved) && saved.length > 0) chats = saved;
    }
  } catch { /* corrupted storage */ }

  if (!chats || chats.length === 0) {
    const legacy = loadLegacyChat();
    if (legacy) chats = [legacy];
  }

  if (!chats || chats.length === 0) {
    const fresh = createChat();
    chats = [fresh];
    activeChatId = fresh.id;
  } else {
    const storedActive = localStorage.getItem(ACTIVE_CHAT_KEY);
    const match = storedActive && chats.find(c => c.id === storedActive);
    activeChatId = match ? match.id : chats[0].id;
  }

  renderChatList();
  renderActiveChat();
}

function loadLegacyChat() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved) || saved.length === 0) return null;
    const chat = createChat(saved);
    localStorage.removeItem(CHAT_STORAGE_KEY);
    return chat;
  } catch {
    return null;
  }
}

function createChat(seedMessages) {
  const id = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const messages = Array.isArray(seedMessages) ? seedMessages : [];
  const title = deriveChatTitle(messages) || 'New chat';
  return { id, title, messages, createdAt: Date.now() };
}

function deriveChatTitle(messages) {
  const firstUser = messages?.find(m => m.role === 'user');
  if (!firstUser || !firstUser.text) return '';
  return formatChatTitle(firstUser.text);
}

function formatChatTitle(text) {
  const clean = text.trim().replace(/\s+/g, ' ');
  if (clean.length <= 40) return clean;
  return clean.slice(0, 37) + '...';
}

function getActiveChat() {
  return chats.find(c => c.id === activeChatId);
}

function setActiveChat(id) {
  if (!id || id === activeChatId) return;
  const target = chats.find(c => c.id === id);
  if (!target) return;
  activeChatId = id;
  saveChatsToStorage();
  renderChatList();
  renderActiveChat();
}

function renderChatList() {
  const list = document.getElementById('chat-list');
  if (!list) return;
  list.innerHTML = chats.map(chat => {
    const count = chat.messages?.length || 0;
    const meta = `${count} message${count === 1 ? '' : 's'}`;
    const active = chat.id === activeChatId ? ' active' : '';
    return `
      <div class="chat-item${active}" data-id="${chat.id}">
        <div class="chat-item-main">
          <span class="chat-item-title">${esc(chat.title || 'New chat')}</span>
          <span class="chat-item-meta">${meta}</span>
        </div>
        <button class="chat-del-btn" data-id="${chat.id}" title="Delete chat" aria-label="Delete chat">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14H6L5 6"/>
            <path d="M10 11v6M14 11v6"/>
          </svg>
        </button>
      </div>`;
  }).join('');
}

function renderActiveChat() {
  const chat = getActiveChat();
  const thread = document.getElementById('thread');
  if (!thread) return;
  thread.innerHTML = '';
  if (!chat || !chat.messages || chat.messages.length === 0) {
    ensureWelcome();
    return;
  }
  document.getElementById('welcome')?.remove();
  for (const msg of chat.messages) {
    if (msg.role === 'user') {
      _appendUserMsgDOM(msg.text);
    } else {
      const { botEl, textEl } = appendBotMsg();
      textEl.innerHTML = renderMarkdown(msg.text);
      textEl.dataset.raw = msg.text;
      if (msg.sources && msg.sources.length > 0) {
        const chips = msg.sources.map(s => `<span class="source-chip">📄 ${esc(s)}</span>`).join('');
        const div = document.createElement('div');
        div.className = 'msg-sources';
        div.innerHTML = chips;
        botEl.querySelector('.bot-body').insertBefore(div, botEl.querySelector('.bot-actions'));
      }
    }
  }
}

function ensureWelcome() {
  if (!document.getElementById('welcome')) {
    const w = buildWelcome();
    document.getElementById('chat-area').insertBefore(w, document.getElementById('thread'));
  }
}

/* Voice INPUT state */
let micRecorder     = null;
let micStream       = null;
let deepgramSocket  = null;
let isRecording     = false;
let voiceTranscript = '';

/* Voice OUTPUT (TTS) state */
let isSpeaking   = false;
let ttsEnabled   = true;
let ttsUtterance = null;

/* Voice MODE state */
let vmActive       = false;
let vmState        = 'idle';  // idle | listening | processing | speaking
let vmRecorder     = null;
let vmStream       = null;
let vmSocket       = null;
let vmFinalText    = '';
let vmSilenceTimer = null;

/* ── Boot ── */
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  loadChatsFromStorage();
  setupSidebar();
  setupUpload();
  setupInput();
  setupButtons();
  setupChatList();
  setupDocPanel();
  setupDocModal();
  setupVoice();
  setupTTS();
  setupVoiceMode();
});

/* ═══════════════════════════════════════════
   STATUS + DOCUMENTS
═══════════════════════════════════════════ */
async function loadStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    document.getElementById('chunk-count').textContent    = data.chunk_count ?? '—';
    document.getElementById('model-name-sidebar').textContent = data.model ?? '—';
    document.getElementById('model-chip-name').textContent    = data.model ?? 'RAG Assistant';
    updateInputState(data.chunk_count > 0);
    await loadDocuments(data.sources || []);
  } catch (e) {
    console.error('Status error', e);
  }
}

async function loadDocuments(statusSources) {
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) {
      // Fallback: use sources from status (plain filenames, no chunk info)
      if (statusSources && statusSources.length > 0) {
        const unique = [...new Set(statusSources)];
        document.getElementById('doc-count').textContent = unique.length;
        renderDocuments(unique.map(name => ({
          name, source: name,
          type: name.split('.').pop().toLowerCase(),
          chunks: '?',
        })));
      }
      return;
    }
    const data = await res.json();
    if (data.error) console.error('Documents API:', data.error);
    const docs = data.documents || [];
    document.getElementById('doc-count').textContent = docs.length || '—';
    renderDocuments(docs);
  } catch (e) {
    console.error('Failed to load documents', e);
  }
}

function renderDocuments(docs) {
  const section = document.getElementById('docs-section');
  const list    = document.getElementById('docs-list');
  if (!docs || docs.length === 0) { section.style.display = 'none'; return; }
  section.style.display = '';
  list.innerHTML = docs.map(doc => `
    <div class="doc-item">
      <span class="doc-item-icon">${fileIcon(doc.name)}</span>
      <div class="doc-item-info">
        <span class="doc-item-name" title="${esc(doc.name)}">${esc(doc.name)}</span>
        <span class="doc-item-meta">${doc.chunks} chunk${doc.chunks !== 1 ? 's' : ''} · ${esc((doc.type||'').toUpperCase())}</span>
      </div>
      <div class="doc-item-btns">
        <button class="doc-btn view-btn" data-source="${esc(doc.source)}" title="View content">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </button>
        <button class="doc-btn del-btn" data-source="${esc(doc.source)}" data-name="${esc(doc.name)}" title="Delete">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
            <path d="M10 11v6M14 11v6"/>
          </svg>
        </button>
      </div>
    </div>`).join('');
}

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
function setupSidebar() {
  const sidebar  = document.getElementById('sidebar');
  const overlay  = document.getElementById('sidebar-overlay');
  const toggleBtn = document.getElementById('sidebar-toggle-btn');
  const closeBtn  = document.getElementById('sidebar-close-btn');

  function openSidebar() {
    sidebar.classList.remove('collapsed');
    sidebar.classList.add('open');
    overlay.classList.add('open');
  }
  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebar.classList.add('collapsed');
    overlay.classList.remove('open');
  }

  toggleBtn.addEventListener('click', () => {
    if (sidebar.classList.contains('collapsed')) openSidebar();
    else closeSidebar();
  });
  closeBtn.addEventListener('click', closeSidebar);
  overlay.addEventListener('click', closeSidebar);
}

/* ═══════════════════════════════════════════
   UPLOAD PANEL
═══════════════════════════════════════════ */
function setupUpload() {
  const zone  = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    addFiles([...e.dataTransfer.files]);
  });
  input.addEventListener('change', () => { addFiles([...input.files]); input.value = ''; });
}

function toggleUploadPanel(open) {
  const panel = document.getElementById('upload-panel');
  uploadPanelOpen = open ?? !uploadPanelOpen;
  panel.style.display = uploadPanelOpen ? '' : 'none';
  document.getElementById('attach-btn').classList.toggle('active', uploadPanelOpen);
}

function addFiles(files) {
  const allowed = new Set(['.pdf', '.txt', '.csv', '.md']);
  files.forEach(f => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.has(ext)) { showToast(`Unsupported: ${f.name}`, 'error'); return; }
    if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
  });
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  const btn  = document.getElementById('ingest-btn');
  list.innerHTML = selectedFiles.map((f, i) => `
    <div class="file-item">
      <span>${fileIcon(f.name)}</span>
      <span class="file-name" title="${esc(f.name)}">${esc(f.name)}</span>
      <button class="remove-file-btn" onclick="removeFile(${i})" title="Remove">&times;</button>
    </div>`).join('');
  btn.disabled = selectedFiles.length === 0;
}

function removeFile(i) { selectedFiles.splice(i, 1); renderFileList(); }

document.getElementById('ingest-btn').addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;
  const btn    = document.getElementById('ingest-btn');
  const status = document.getElementById('ingest-status');
  btn.disabled    = true;
  btn.textContent = 'Ingesting…';
  status.textContent = '';

  const form = new FormData();
  selectedFiles.forEach(f => form.append('files', f));

  try {
    const res  = await fetch('/api/ingest', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Ingest failed');
    status.textContent = data.message;
    status.style.color = 'var(--accent)';
    selectedFiles = [];
    renderFileList();
    await loadStatus();
    showToast(data.message, 'success');
    document.getElementById('welcome')?.remove();
    setTimeout(() => toggleUploadPanel(false), 1200);
  } catch (e) {
    status.textContent = e.message;
    status.style.color = 'var(--danger)';
    showToast(e.message, 'error');
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Ingest Documents';
  }
});

document.getElementById('cancel-upload-btn').addEventListener('click', () => toggleUploadPanel(false));

/* ═══════════════════════════════════════════
   INPUT + SEND
═══════════════════════════════════════════ */
function setupInput() {
  const ta  = document.getElementById('question-input');
  const btn = document.getElementById('send-btn');

  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
    btn.disabled = ta.value.trim() === '' || ta.disabled || isStreaming;
  });

  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  btn.addEventListener('click', sendMessage);
}

function updateInputState(enabled) {
  const ta  = document.getElementById('question-input');
  const btn = document.getElementById('send-btn');
  ta.disabled  = !enabled;
  btn.disabled = !enabled;
  ta.placeholder = enabled ? 'Message RAG Assistant…' : 'Upload and ingest documents first…';
}

async function sendMessage() {
  const ta  = document.getElementById('question-input');
  const btn = document.getElementById('send-btn');
  const question = ta.value.trim();
  if (!question || isStreaming) return;

  stopSpeaking();
  document.getElementById('welcome')?.remove();
  isStreaming = true;

  appendUserMsg(question);
  ta.value = '';
  ta.style.height = 'auto';
  btn.disabled = true;

  const { botEl, textEl } = appendBotMsg();

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    await streamText(textEl, data.answer);
    speakResponse(data.answer);

    if (data.sources && data.sources.length > 0) {
      const chips = data.sources.map(s =>
        `<span class="source-chip">📄 ${esc(s)}</span>`).join('');
      const div = document.createElement('div');
      div.className = 'msg-sources';
      div.innerHTML = chips;
      botEl.querySelector('.bot-body').insertBefore(div, botEl.querySelector('.bot-actions'));
    }

    const chat = getActiveChat();
    if (chat) {
      chat.messages.push({ role: 'bot', text: data.answer, sources: data.sources || [] });
      saveChatsToStorage();
      renderChatList();
    }
  } catch (e) {
    textEl.innerHTML = `<span style="color:var(--danger)">Error: ${esc(e.message)}</span>`;
  } finally {
    isStreaming = false;
    btn.disabled = ta.value.trim() === '' || ta.disabled;
  }
}

function useSuggestion(el) {
  const ta = document.getElementById('question-input');
  ta.value = el.querySelector('span').textContent;
  ta.dispatchEvent(new Event('input'));
  sendMessage();
}

/* ═══════════════════════════════════════════
   VOICE INPUT — Web Speech API
═══════════════════════════════════════════ */
let _speechRecognition = null;

function setupVoice() {
  document.getElementById('mic-btn').addEventListener('click', () => {
    if (isRecording) stopRecording();
    else startRecording();
  });
  document.getElementById('voice-stop-btn').addEventListener('click', stopRecording);
}

function startRecording() {
  if (isRecording) return;
  stopSpeaking();

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast('Voice input not supported in this browser. Use Chrome or Edge.', 'error');
    return;
  }

  const langVal = document.getElementById('voice-lang')?.value || 'auto';

  _speechRecognition = new SpeechRecognition();
  _speechRecognition.continuous      = true;
  _speechRecognition.interimResults  = true;
  _speechRecognition.lang            = langVal === 'auto' ? '' : langVal;

  isRecording     = true;
  voiceTranscript = '';
  setRecordingUI(true);

  _speechRecognition.onresult = e => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) {
        voiceTranscript += (voiceTranscript ? ' ' : '') + t;
      } else {
        interim = t;
      }
    }
    const ta = document.getElementById('question-input');
    ta.value = voiceTranscript + (interim ? (voiceTranscript ? ' ' : '') + interim : '');
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
    if (interim) {
      document.getElementById('voice-label').textContent = `Listening… "${interim}"`;
    }
  };

  _speechRecognition.onerror = e => {
    if (e.error !== 'aborted') showToast(`Voice error: ${e.error}`, 'error');
    stopRecording();
  };

  _speechRecognition.onend = () => {
    if (isRecording) stopRecording();
  };

  _speechRecognition.start();
}

function stopRecording() {
  isRecording = false;
  setRecordingUI(false);

  if (_speechRecognition) {
    try { _speechRecognition.stop(); } catch {}
    _speechRecognition = null;
  }

  voiceTranscript = '';

  const ta = document.getElementById('question-input');
  if (ta.value.trim()) {
    document.getElementById('send-btn').disabled = false;
    ta.focus();
  }
}

function setRecordingUI(active) {
  const micBtn      = document.getElementById('mic-btn');
  const voiceStatus = document.getElementById('voice-status');
  const voiceLabel  = document.getElementById('voice-label');

  micBtn.classList.toggle('recording', active);
  voiceStatus.style.display = active ? 'flex' : 'none';
  if (active) voiceLabel.textContent = 'Listening…';
}

function getSupportedMime() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

/* ═══════════════════════════════════════════
   VOICE OUTPUT — TTS (Web Speech API)
═══════════════════════════════════════════ */
function setupTTS() {
  const stopBtn = document.getElementById('tts-stop-btn');
  if (stopBtn) stopBtn.addEventListener('click', stopSpeaking);
}

function stripMarkdownForTTS(text) {
  return text
    .replace(/```[\s\S]*?```/g, 'code block. ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*\*(.+?)\*\*\*/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*([^*\n]+)\*/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^[-*+]\s+/gm, '')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/^>\s+/gm, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/---/g, '')
    .replace(/\n{2,}/g, '. ')
    .replace(/\n/g, ' ')
    .trim();
}

function speakResponse(text) {
  if (!ttsEnabled || !('speechSynthesis' in window)) return;
  stopSpeaking();

  const clean = stripMarkdownForTTS(text);
  if (!clean) return;

  ttsUtterance = new SpeechSynthesisUtterance(clean);
  ttsUtterance.rate  = 1.0;
  ttsUtterance.pitch = 1.0;

  ttsUtterance.onstart = () => {
    isSpeaking = true;
    setSpeakingUI(true);
  };
  ttsUtterance.onend = () => {
    isSpeaking = false;
    setSpeakingUI(false);
  };
  ttsUtterance.onerror = () => {
    isSpeaking = false;
    setSpeakingUI(false);
  };

  window.speechSynthesis.speak(ttsUtterance);
}

function stopSpeaking() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  isSpeaking = false;
  ttsUtterance = null;
  setSpeakingUI(false);
}

function setSpeakingUI(active) {
  const ttsStatus = document.getElementById('tts-status');
  if (ttsStatus) ttsStatus.style.display = active ? 'flex' : 'none';
}

/* ═══════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════ */
function setupButtons() {
  /* Attach / upload trigger */
  document.getElementById('attach-btn').addEventListener('click', () => toggleUploadPanel());
  document.getElementById('upload-trigger-btn').addEventListener('click', () => {
    toggleUploadPanel(true);
    document.getElementById('attach-btn').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  /* New chat */
  document.getElementById('new-chat-btn').addEventListener('click', () => {
    stopSpeaking();
    const fresh = createChat();
    chats.unshift(fresh);
    activeChatId = fresh.id;
    saveChatsToStorage();
    renderChatList();
    renderActiveChat();
    const ta = document.getElementById('question-input');
    ta.value = '';
    ta.style.height = 'auto';
    document.getElementById('send-btn').disabled = true;
    fetch('/api/clear-memory', { method: 'POST' });
  });

  /* Clear memory */
  document.getElementById('clear-memory-btn').addEventListener('click', async () => {
    await fetch('/api/clear-memory', { method: 'POST' });
    showToast('Conversation memory cleared', 'info');
  });

  /* Reset DB */
  document.getElementById('reset-btn').addEventListener('click', async () => {
    if (!confirm('Remove all documents from the knowledge base? This cannot be undone.')) return;
    await fetch('/api/reset', { method: 'POST' });
    await loadStatus();
    document.getElementById('thread').innerHTML = '';
    showToast('Knowledge base cleared', 'info');
  });
}

function setupChatList() {
  const list = document.getElementById('chat-list');
  if (!list) return;
  list.addEventListener('click', e => {
    const del = e.target.closest('.chat-del-btn');
    if (del) {
      e.stopPropagation();
      deleteChat(del.dataset.id);
      return;
    }
    const item = e.target.closest('.chat-item');
    if (!item) return;
    setActiveChat(item.dataset.id);
  });
}

function deleteChat(chatId) {
  const idx = chats.findIndex(c => c.id === chatId);
  if (idx === -1) return;
  const target = chats[idx];
  if (!confirm(`Delete "${target.title || 'New chat'}"? This cannot be undone.`)) return;

  chats.splice(idx, 1);

  if (chats.length === 0) {
    const fresh = createChat();
    chats = [fresh];
    activeChatId = fresh.id;
  } else if (activeChatId === chatId) {
    const next = chats[idx] || chats[idx - 1] || chats[0];
    activeChatId = next.id;
  }

  saveChatsToStorage();
  renderChatList();
  renderActiveChat();
}

function buildWelcome() {
  const div = document.createElement('div');
  div.id = 'welcome';
  div.className = 'welcome';
  div.innerHTML = `
    <div class="welcome-logo">
      <div class="welcome-glow"></div>
      <span class="welcome-initial">R</span>
    </div>
    <h1 class="welcome-title">How can I help you?</h1>
    <p class="welcome-sub">Ask me anything about your uploaded documents.</p>
    <div class="suggestion-grid">
      <button class="suggestion-card" onclick="useSuggestion(this)">
        <div class="suggestion-icon">🔎</div><span>What is RAG and why does it reduce hallucinations?</span>
      </button>
      <button class="suggestion-card" onclick="useSuggestion(this)">
        <div class="suggestion-icon">🧠</div><span>What are the three types of machine learning?</span>
      </button>
      <button class="suggestion-card" onclick="useSuggestion(this)">
        <div class="suggestion-icon">⚡</div><span>Explain the transformer attention mechanism</span>
      </button>
      <button class="suggestion-card" onclick="useSuggestion(this)">
        <div class="suggestion-icon">📋</div><span>Summarize the key points from my documents</span>
      </button>
    </div>`;
  return div;
}

/* ═══════════════════════════════════════════
   DOCUMENT PANEL
═══════════════════════════════════════════ */
function setupDocPanel() {
  document.getElementById('docs-list').addEventListener('click', e => {
    const view = e.target.closest('.view-btn');
    const del  = e.target.closest('.del-btn');
    if (view) openDocModal(view.dataset.source);
    if (del)  deleteDocument(del.dataset.source, del.dataset.name);
  });

  document.getElementById('refresh-docs-btn').addEventListener('click', loadStatus);
}

async function deleteDocument(source, name) {
  if (!confirm(`Delete "${name}" from the knowledge base?`)) return;
  try {
    const res  = await fetch(`/api/documents?source=${encodeURIComponent(source)}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Delete failed');
    showToast(data.message, 'success');
    await loadStatus();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

/* ═══════════════════════════════════════════
   DOCUMENT VIEWER MODAL
═══════════════════════════════════════════ */
function setupDocModal() {
  document.getElementById('doc-modal-close').addEventListener('click', closeDocModal);
  document.getElementById('doc-modal-close-foot').addEventListener('click', closeDocModal);
  document.getElementById('doc-overlay').addEventListener('click', e => {
    if (e.target.id === 'doc-overlay') closeDocModal();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDocModal(); });
}

async function openDocModal(source) {
  const overlay  = document.getElementById('doc-overlay');
  const body     = document.getElementById('doc-modal-body');
  const name     = document.getElementById('doc-modal-name');
  const info     = document.getElementById('doc-modal-info');
  const icon     = document.getElementById('doc-modal-icon');
  const badge    = document.getElementById('doc-type-badge');
  const foot     = document.getElementById('doc-modal-foot');
  const trunc    = document.getElementById('doc-trunc');

  body.innerHTML = `<div class="doc-loading"><div class="dots"><span></span><span></span><span></span></div> Loading…</div>`;
  foot.style.display = 'none';
  name.textContent = source.split(/[\\/]/).pop();
  info.textContent = ''; badge.textContent = '…'; icon.textContent = '📄';
  overlay.classList.add('open');

  try {
    const res  = await fetch(`/api/documents/content?source=${encodeURIComponent(source)}`);
    const data = await res.json();
    if (data.error) { body.innerHTML = `<div class="doc-loading" style="color:var(--danger)">${esc(data.error)}</div>`; return; }

    name.textContent  = data.name;
    badge.textContent = data.type.toUpperCase();
    icon.textContent  = fileIcon(data.name);
    info.textContent  = `${data.chunks} chunks · ${data.total_chars.toLocaleString()} characters`;

    const pre = document.createElement('pre');
    pre.className   = 'doc-content';
    pre.textContent = data.content;
    body.innerHTML  = '';
    body.appendChild(pre);

    foot.style.display = 'flex';
    trunc.textContent  = data.truncated
      ? `Showing first 15,000 of ${data.total_chars.toLocaleString()} characters`
      : '';
  } catch (e) {
    body.innerHTML = `<div class="doc-loading" style="color:var(--danger)">Failed to load content.</div>`;
  }
}

function closeDocModal() {
  document.getElementById('doc-overlay').classList.remove('open');
}

/* ═══════════════════════════════════════════
   MESSAGE RENDERING
═══════════════════════════════════════════ */
function _appendUserMsgDOM(text) {
  const thread = document.getElementById('thread');
  const div    = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  thread.appendChild(div);
  scrollBottom();
}

function appendUserMsg(text) {
  _appendUserMsgDOM(text);
  const chat = getActiveChat();
  if (!chat) return;
  chat.messages.push({ role: 'user', text });
  if (!chat.title || chat.title === 'New chat') {
    chat.title = formatChatTitle(text) || 'New chat';
  }
  saveChatsToStorage();
  renderChatList();
}

function appendBotMsg() {
  const thread = document.getElementById('thread');
  const div    = document.createElement('div');
  div.className = 'msg msg-bot';
  div.innerHTML = `
    <div class="bot-avatar">R</div>
    <div class="bot-body">
      <div class="bot-text"><div class="dots"><span></span><span></span><span></span></div></div>
      <div class="bot-actions">
        <button class="act-btn copy-btn" title="Copy">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2"/>
            <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
          </svg>
          Copy
        </button>
      </div>
    </div>`;
  thread.appendChild(div);
  scrollBottom();

  const textEl = div.querySelector('.bot-text');
  div.querySelector('.copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(textEl.dataset.raw || textEl.textContent)
      .then(() => showToast('Copied!', 'success'));
  });

  return { botEl: div, textEl };
}

/* ── Streaming text effect ── */
async function streamText(el, rawText) {
  el.querySelector('.dots')?.remove();
  el.dataset.raw = rawText;

  const words   = rawText.split(/(\s+)/);
  let revealed  = '';

  for (const chunk of words) {
    revealed += chunk;
    el.innerHTML = esc(revealed) + '<span class="cursor"></span>';
    scrollBottom();
    await sleep(16 + Math.random() * 18);
  }

  el.innerHTML = renderMarkdown(rawText);
  scrollBottom();
}

/* ── Markdown renderer ── */
function renderMarkdown(raw) {
  let t = raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  /* Fenced code blocks */
  const blocks = [];
  t = t.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const label = lang.trim() || 'text';
    const idx   = blocks.length;
    blocks.push(
      `<div class="code-block">` +
      `<div class="code-header"><span class="code-lang">${label}</span>` +
      `<button class="copy-code" onclick="copyCode(this)">Copy</button></div>` +
      `<pre>${code.trim()}</pre></div>`
    );
    return `\x00BLOCK${idx}\x00`;
  });

  /* Inline code */
  t = t.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');

  /* Bold / Italic */
  t = t.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  t = t.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  t = t.replace(/\*([^*\n]+)\*/g,     '<em>$1</em>');

  /* Blockquote */
  t = t.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  /* Horizontal rule */
  t = t.replace(/^---$/gm, '<hr>');

  /* Headers */
  t = t.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  t = t.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  t = t.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  /* Unordered lists */
  t = t.replace(/((?:^[-*+] .+\n?)+)/gm, m => {
    const items = m.trim().split('\n').map(l => `<li>${l.replace(/^[-*+] /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });

  /* Ordered lists */
  t = t.replace(/((?:^\d+\. .+\n?)+)/gm, m => {
    const items = m.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  /* Paragraphs */
  t = t.split(/\n\n+/).map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<(h[1-3]|ul|ol|div|blockquote|hr)/.test(p)) return p;
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('');

  /* Restore code blocks */
  blocks.forEach((b, i) => { t = t.replace(`\x00BLOCK${i}\x00`, b); });

  return t;
}

function copyCode(btn) {
  const code = btn.closest('.code-block').querySelector('pre').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  });
}

/* ═══════════════════════════════════════════
   VOICE MODE — GPT-style conversation
═══════════════════════════════════════════ */
function setupVoiceMode() {
  const openBtn = document.getElementById('vm-open-btn');
  if (!openBtn) return;
  openBtn.addEventListener('click', openVoiceMode);
  document.getElementById('vm-end-btn').addEventListener('click', closeVoiceMode);
  document.getElementById('vm-orb').addEventListener('click', () => {
    if (vmState === 'idle') vmStartListening();
    else if (vmState === 'listening') vmStopAndProcess();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && vmActive) closeVoiceMode();
  });
}

async function openVoiceMode() {
  stopSpeaking();
  const overlay = document.getElementById('voice-mode-overlay');
  overlay.classList.add('open');
  vmActive = true;
  document.getElementById('vm-transcript').textContent = '';
  document.getElementById('vm-response').textContent   = '';
  setVMState('idle');
  await vmStartListening();
}

function closeVoiceMode() {
  vmActive = false;
  if (vmSilenceTimer) { clearTimeout(vmSilenceTimer); vmSilenceTimer = null; }
  vmCleanupMic();
  stopSpeaking();
  document.getElementById('voice-mode-overlay').classList.remove('open');
  setVMState('idle');
  document.getElementById('vm-transcript').textContent = '';
  document.getElementById('vm-response').textContent   = '';
}

function setVMState(state) {
  vmState = state;
  const orb = document.getElementById('vm-orb');
  if (orb) orb.className = `vm-orb vm-orb-${state}`;
  const labels = { idle: 'Tap orb to speak', listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' };
  const el = document.getElementById('vm-status');
  if (el) el.textContent = labels[state] ?? '';
}

function vmStartListening() {
  if (!vmActive || vmState === 'listening') return;

  stopSpeaking();
  vmFinalText = '';
  document.getElementById('vm-transcript').textContent = '';

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    document.getElementById('vm-status').textContent = 'Voice not supported. Use Chrome or Edge.';
    return;
  }

  vmSocket = new SpeechRecognition();
  vmSocket.continuous     = true;
  vmSocket.interimResults = true;
  vmSocket.lang           = '';

  setVMState('listening');

  vmSocket.onresult = e => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) {
        vmFinalText += (vmFinalText ? ' ' : '') + t;
        document.getElementById('vm-transcript').textContent = vmFinalText;
        if (vmSilenceTimer) clearTimeout(vmSilenceTimer);
        vmSilenceTimer = setTimeout(() => {
          if (vmState === 'listening' && vmFinalText.trim()) vmStopAndProcess();
        }, 1600);
      } else {
        interim = t;
        document.getElementById('vm-transcript').textContent =
          (vmFinalText ? vmFinalText + ' ' : '') + interim;
      }
    }
  };

  vmSocket.onerror = () => { if (vmState === 'listening') setVMState('idle'); };
  vmSocket.onend   = () => { if (vmState === 'listening') setVMState('idle'); };

  vmSocket.start();
}

function vmStopAndProcess() {
  if (vmSilenceTimer) { clearTimeout(vmSilenceTimer); vmSilenceTimer = null; }
  vmCleanupMic();
  const text = vmFinalText.trim();
  if (text) vmProcessQuery(text);
  else if (vmActive) vmStartListening();
}

function vmCleanupMic() {
  if (vmSocket) {
    try { vmSocket.stop(); } catch {}
    vmSocket = null;
  }
  vmRecorder = null;
  if (vmStream) { vmStream.getTracks().forEach(t => t.stop()); vmStream = null; }
}

async function vmProcessQuery(text) {
  if (!vmActive) return;
  setVMState('processing');
  document.getElementById('vm-transcript').textContent = text;
  document.getElementById('vm-response').textContent   = '';

  appendUserMsg(text);
  document.getElementById('welcome')?.remove();
  const { botEl, textEl } = appendBotMsg();

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    await streamText(textEl, data.answer);
    if (data.sources?.length > 0) {
      const chips = data.sources.map(s => `<span class="source-chip">📄 ${esc(s)}</span>`).join('');
      const div = document.createElement('div');
      div.className = 'msg-sources';
      div.innerHTML = chips;
      botEl.querySelector('.bot-body').insertBefore(div, botEl.querySelector('.bot-actions'));
    }
    const chat = getActiveChat();
    if (chat) {
      chat.messages.push({ role: 'bot', text: data.answer, sources: data.sources || [] });
      saveChatsToStorage();
      renderChatList();
    }

    const preview = stripMarkdownForTTS(data.answer);
    document.getElementById('vm-response').textContent =
      preview.length > 130 ? preview.slice(0, 130) + '…' : preview;

    if (vmActive) vmSpeakThenListen(data.answer);
  } catch (e) {
    document.getElementById('vm-status').textContent = 'Error: ' + e.message;
    setTimeout(() => { if (vmActive) vmStartListening(); }, 2500);
  }
}

function vmSpeakThenListen(text) {
  if (!vmActive || !('speechSynthesis' in window)) {
    if (vmActive) setTimeout(() => vmStartListening(), 600);
    return;
  }
  window.speechSynthesis.cancel();
  setVMState('speaking');

  const utt  = new SpeechSynthesisUtterance(stripMarkdownForTTS(text));
  utt.rate   = 1.05;
  utt.pitch  = 1.0;
  utt.onend  = utt.onerror = () => {
    if (vmActive) setTimeout(() => vmStartListening(), 700);
  };
  window.speechSynthesis.speak(utt);
}

/* ═══════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════ */
function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  return { pdf: '📄', txt: '📝', csv: '📊', md: '📋' }[ext] || '📄';
}

function scrollBottom() {
  const a = document.getElementById('chat-area');
  a.scrollTop = a.scrollHeight;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = `toast show ${type}`;
  setTimeout(() => { t.className = 'toast'; }, 3400);
}
