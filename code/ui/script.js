// The page talks to the FastAPI backend with fetch():
//   POST /upload  -> send a PDF
//   POST /ask     -> send a question, get { answer, sources, steps }

const messages = document.getElementById("messages");
const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question-input");
const uploadButton = document.getElementById("upload-button");
const pdfInput = document.getElementById("pdf-input");
const uploadStatus = document.getElementById("upload-status");


// Send a request to the API. Returns the JSON data, or throws an Error with a readable message.
async function callApi(endpoint, options) {
  const response = await fetch(endpoint, { method: "POST", ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Something went wrong");
  return data;
}


// Add a chat bubble. Returns it so we can change it later (e.g. replace "Thinking...").
function addMessage(role, html) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.innerHTML = html;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;  // scroll to the newest message
  return bubble;
}


// Make text safe to put inside HTML (so "<script>" shows as text, not code)
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}


// Markdown from Gemini -> safe HTML
function markdownToHtml(text) {
  return DOMPurify.sanitize(marked.parse(text));
}


// Build the answer bubble: agent steps + answer + collapsible sources
function answerHtml(data) {
  let html = `<div class="steps">🤖 Agents: ${escapeHtml(data.steps.join(" → "))}</div>`;
  html += markdownToHtml(data.answer);

  if (data.sources.length > 0) {
    html += `<details><summary>📚 Sources (${data.sources.length})</summary>`;
    for (const s of data.sources) {
      html += `<div class="source">
        <strong>[${s.id}] ${escapeHtml(s.source)} — page ${s.page}</strong> (relevance ${s.score})
        <div class="source-text">${escapeHtml(s.text.slice(0, 400))}...</div>
      </div>`;
    }
    html += `</details>`;
  }
  return html;
}


// ---------- Upload a PDF ----------
uploadButton.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a PDF first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadButton.disabled = true;
  uploadStatus.textContent = "Reading and indexing... this can take a minute.";
  try {
    const data = await callApi("/upload", { body: formData });
    uploadStatus.textContent = `✅ Added ${data.file} (${data.chunks} chunks)`;
  } catch (error) {
    uploadStatus.textContent = `❌ ${error.message}`;
  }
  uploadButton.disabled = false;
});


// ---------- Ask a question ----------
questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();  // stop the browser from reloading the page
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", escapeHtml(question));
  questionInput.value = "";
  const bubble = addMessage("assistant", "Thinking...");

  const askButton = questionForm.querySelector("button");
  askButton.disabled = true;
  try {
    const data = await callApi("/ask", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    bubble.innerHTML = answerHtml(data);
  } catch (error) {
    bubble.classList.add("error");
    bubble.textContent = `❌ ${error.message}`;
  }
  askButton.disabled = false;
  messages.scrollTop = messages.scrollHeight;
});
