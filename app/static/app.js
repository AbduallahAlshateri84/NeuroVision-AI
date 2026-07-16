const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const selectedFile = document.getElementById("selectedFile");
const localPreview = document.getElementById("localPreview");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const analyseButton = document.getElementById("analyseButton");
const errorMessage = document.getElementById("errorMessage");
const serverStatus = document.getElementById("serverStatus");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const predictedLabel = document.getElementById("predictedLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceRing = document.getElementById("confidenceRing");
const resultOriginal = document.getElementById("resultOriginal");
const resultHeatmap = document.getElementById("resultHeatmap");
const probabilityBars = document.getElementById("probabilityBars");

let activeFile = null;

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

function setFile(file) {
  clearError();
  const acceptedTypes = ["image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"];
  if (!acceptedTypes.includes(file.type)) {
    showError("Please choose a PNG, JPEG, WEBP, BMP, or TIFF image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("The image is larger than the 10 MB upload limit.");
    return;
  }

  activeFile = file;
  localPreview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  selectedFile.classList.remove("hidden");
  analyseButton.disabled = false;
}

function resetFile() {
  activeFile = null;
  fileInput.value = "";
  localPreview.removeAttribute("src");
  selectedFile.classList.add("hidden");
  analyseButton.disabled = true;
  clearError();
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    serverStatus.classList.remove("checking", "ready", "error");
    serverStatus.classList.add(data.model_ready ? "ready" : "error");
    serverStatus.querySelector("span:last-child").textContent = data.model_ready
      ? "Model online"
      : "Model file needed";
    serverStatus.title = data.message;
  } catch {
    serverStatus.classList.remove("checking", "ready");
    serverStatus.classList.add("error");
    serverStatus.querySelector("span:last-child").textContent = "Server unavailable";
  }
}

function renderProbabilities(rows) {
  probabilityBars.innerHTML = "";
  rows.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "probability-row";
    wrapper.innerHTML = `
      <span class="probability-label">${row.label}</span>
      <div class="probability-track" aria-label="${row.label}: ${row.percentage}%">
        <div class="probability-fill"></div>
      </div>
      <span class="probability-value">${row.percentage.toFixed(2)}%</span>
    `;
    probabilityBars.appendChild(wrapper);
    requestAnimationFrame(() => {
      wrapper.querySelector(".probability-fill").style.width = `${row.percentage}%`;
    });
  });
}

function renderResult(data) {
  predictedLabel.textContent = data.prediction.label;
  confidenceValue.textContent = `${data.prediction.confidence_percentage.toFixed(1)}%`;
  confidenceRing.style.setProperty(
    "--confidence",
    `${data.prediction.confidence_percentage * 3.6}deg`
  );
  resultOriginal.src = data.images.original;
  resultHeatmap.src = data.images.gradcam;
  renderProbabilities(data.probabilities);
  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");
}

async function analyseImage() {
  if (!activeFile) return;
  clearError();
  analyseButton.disabled = true;
  analyseButton.classList.add("loading");
  analyseButton.querySelector(".button-text").textContent = "Analysing CNN features";

  const formData = new FormData();
  formData.append("file", activeFile);

  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The server could not analyse this image.");
    }
    renderResult(payload);
  } catch (error) {
    showError(error.message || "Prediction failed. Please try again.");
  } finally {
    analyseButton.disabled = !activeFile;
    analyseButton.classList.remove("loading");
    analyseButton.querySelector(".button-text").textContent = "Analyse MRI";
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) setFile(fileInput.files[0]);
});
removeFile.addEventListener("click", resetFile);
analyseButton.addEventListener("click", analyseImage);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files?.[0]) setFile(event.dataTransfer.files[0]);
});

checkHealth();
