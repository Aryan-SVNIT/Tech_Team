const audioForm = document.querySelector('#audio-form');
const MAX_UPLOAD_BYTES = Number(audioForm.dataset.maxUploadBytes) || 100 * 1024 * 1024;
const MAX_UPLOAD_MB = Number(audioForm.dataset.maxUploadMb) || 100;
const audioInput = document.querySelector('#audio-file');
const selectedFileLabel = document.querySelector('#selected-file');
const transcribeButton = document.querySelector('#transcribe-button');
const languageSelect = document.querySelector('#language-select');
const languageSelectionStatus = document.querySelector('#language-selection-status');
const recordButton = document.querySelector('#record-button');
const stopButton = document.querySelector('#stop-button');
const recordHint = document.querySelector('#record-hint');
const recordingTime = document.querySelector('#recording-time');
const notice = document.querySelector('#notice');
const noticeText = document.querySelector('#notice-text');
const transcriptCard = document.querySelector('#transcript-card');
const transcriptText = document.querySelector('#transcript-text');
const downloadButton = document.querySelector('#download-button');

let recorder = null;
let audioStream = null;
let recordedChunks = [];
let timerHandle = null;
let recordingStartedAt = 0;
let busy = false;

const LANGUAGE_NAMES = { auto: 'Auto-detect', en: 'English', hi: 'Hindi', gu: 'Gujarati' };

function updateLanguageSelectionStatus() {
  const selectedLanguage = languageSelect.value;
  const languageName = LANGUAGE_NAMES[selectedLanguage] || selectedLanguage;
  languageSelectionStatus.textContent = selectedLanguage === 'auto'
    ? 'Selected: Auto-detect. Whisper will choose a language.'
    : `Selected: ${languageName} (${selectedLanguage}).`;
}

languageSelect.addEventListener('change', updateLanguageSelectionStatus);
updateLanguageSelectionStatus();

function showNotice(message, type = 'success') {
  noticeText.textContent = message;
  notice.className = `notice ${type}`;
  notice.hidden = false;
}

function hideNotice() {
  notice.hidden = true;
}

function setBusy(value) {
  busy = value;
  audioInput.disabled = value;
  languageSelect.disabled = value;
  transcribeButton.disabled = value || audioInput.files.length === 0;
  recordButton.disabled = value;
  stopButton.disabled = value;
  if (value) {
    recordHint.textContent = 'Transcribing locally. Longer recordings can take a little while.';
  } else if (!recorder || recorder.state !== 'recording') {
    recordHint.textContent = 'Microphone access is requested by your browser.';
  }
}

audioInput.addEventListener('change', () => {
  const file = audioInput.files[0];
  selectedFileLabel.textContent = file ? file.name : 'No file selected';
  transcribeButton.disabled = busy || !file;
  hideNotice();
});

audioForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = audioInput.files[0];
  if (!file) {
    showNotice('Choose an audio file first.', 'error');
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    showNotice(`That file is over the ${MAX_UPLOAD_MB} MB limit. Choose a smaller audio file.`, 'error');
    return;
  }
  await transcribeAudio(file, file.name);
});

async function transcribeAudio(audio, filename) {
  if (busy) return;
  if (audio.size > MAX_UPLOAD_BYTES) {
    showNotice(`That recording is over the ${MAX_UPLOAD_MB} MB limit. Shorten it or increase MAX_UPLOAD_MB before starting the app.`, 'error');
    return;
  }
  // Both uploaded Files and recorded Blobs pass through this helper so the
  // currently selected language always accompanies either kind of audio.
  const selectedLanguage = languageSelect.value;
  hideNotice();
  transcriptCard.hidden = true;
  downloadButton.disabled = true;
  setBusy(true);
  showNotice('Preparing your audio for transcription…', 'loading');

  const formData = new FormData();
  formData.append('file', audio, filename);
  formData.append('language', selectedLanguage);

  try {
    const response = await fetch('/transcribe', { method: 'POST', body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || 'The transcription request could not be completed.');
    }

    const transcript = payload.transcript || '';
    transcriptText.textContent = transcript;
    transcriptCard.hidden = false;
    downloadButton.disabled = transcript.length === 0;

    const acceptedLanguage = payload.selected_language || selectedLanguage;
    if (acceptedLanguage === 'auto') {
      const detectedCode = payload.detected_language;
      const detectedName = detectedCode
        ? ({ ...LANGUAGE_NAMES, ur: 'Urdu' }[detectedCode] || detectedCode)
        : null;
      const languageNote = detectedName
        ? ` Language detected: ${detectedName} (${detectedCode}).`
        : '';
      showNotice(`${payload.message || 'Transcription complete.'}${languageNote}`, 'success');
    } else {
      const languageName = LANGUAGE_NAMES[acceptedLanguage] || acceptedLanguage;
      showNotice(`${payload.message || 'Transcription complete.'} Selected: ${languageName} (${acceptedLanguage}).`, 'success');
    }
  } catch (error) {
    showNotice(error.message || 'Something went wrong while transcribing this audio.', 'error');
  } finally {
    setBusy(false);
  }
}

function updateRecordingClock() {
  const seconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
  const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
  const remainder = String(seconds % 60).padStart(2, '0');
  recordingTime.textContent = `${minutes}:${remainder}`;
}

function recordingExtension(mimeType) {
  const mediaType = (mimeType || '').split(';', 1)[0].trim().toLowerCase();
  const extensions = {
    'audio/webm': 'webm',
    'video/webm': 'webm',
    'audio/ogg': 'ogg',
    'application/ogg': 'ogg',
    'audio/mp4': 'm4a',
    'video/mp4': 'm4a',
    'audio/x-m4a': 'm4a',
    'audio/wav': 'wav',
    'audio/wave': 'wav',
    'audio/x-wav': 'wav',
    'audio/mpeg': 'mp3',
  };
  return extensions[mediaType] || null;
}

function stopAudioTracks() {
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
}

recordButton.addEventListener('click', async () => {
  hideNotice();
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showNotice('Microphone recording is not supported by this browser. Try a current version of Chrome, Edge, or Firefox.', 'error');
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const supportedTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    const mimeType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type));
    recorder = mimeType
      ? new MediaRecorder(audioStream, { mimeType })
      : new MediaRecorder(audioStream);
    recordedChunks = [];

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) recordedChunks.push(event.data);
    });
    recorder.addEventListener('error', () => {
      stopAudioTracks();
      clearInterval(timerHandle);
      recordingTime.hidden = true;
      recordButton.hidden = false;
      stopButton.hidden = true;
      setBusy(false);
      showNotice('The browser could not finish recording. Check your microphone and try again.', 'error');
    }, { once: true });
    recorder.addEventListener('stop', async () => {
      stopAudioTracks();
      clearInterval(timerHandle);
      recordingTime.hidden = true;
      recordButton.hidden = false;
      stopButton.hidden = true;
      audioInput.disabled = false;
      languageSelect.disabled = false;
      recordButton.disabled = false;

      const blobType = recorder.mimeType || recordedChunks.find((chunk) => chunk.type)?.type || '';
      const extension = recordingExtension(blobType);
      if (!extension) {
        setBusy(false);
        showNotice(
          `The browser reported an unsupported recording format (${blobType || 'unknown'}). Try uploading the recording instead.`,
          'error',
        );
        return;
      }
      const audio = new Blob(recordedChunks, { type: blobType });
      if (audio.size === 0) {
        setBusy(false);
        showNotice('No audio was captured. Check your microphone and record again.', 'error');
        return;
      }
      await transcribeAudio(audio, `recording.${extension}`);
    }, { once: true });

    recorder.start(250);
    recordingStartedAt = Date.now();
    recordingTime.textContent = '00:00';
    recordingTime.hidden = false;
    timerHandle = setInterval(updateRecordingClock, 500);
    recordButton.hidden = true;
    stopButton.hidden = false;
    audioInput.disabled = true;
    languageSelect.disabled = true;
    transcribeButton.disabled = true;
    recordHint.textContent = 'Recording… select Stop & transcribe when you’re done.';
  } catch (error) {
    stopAudioTracks();
    const permissionDenied = error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError';
    showNotice(
      permissionDenied
        ? 'Microphone permission was denied. Allow access in your browser settings, then try again.'
        : 'Could not start the microphone. Check that it is connected and available.',
      'error',
    );
  }
});

stopButton.addEventListener('click', () => {
  if (recorder?.state === 'recording') {
    stopButton.disabled = true;
    recorder.stop();
  }
});

downloadButton.addEventListener('click', () => {
  const transcript = transcriptText.textContent;
  if (!transcript) return;
  const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'transcript.txt';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
});
