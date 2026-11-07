// This component never sends recordings or events to the Python server.
export default function ({ parentElement, data = {} }) {
  parentElement.querySelector('.pdf-actions').innerHTML = data.pdf_controls || '';
  const printButton = parentElement.querySelector('[data-action="print"]');
  const recordButton = parentElement.querySelector('[data-action="record"]');
  const feedback = parentElement.querySelector('.feedback');
  const download = parentElement.querySelector('.download');
  const recordTooltip = parentElement.querySelector('[data-tooltip="record"]');
  const controls = [...parentElement.querySelectorAll('.action-control')];
  const tooltipListeners = [];
  let recorder = null;
  let stream = null;
  let disposed = false;
  let downloadUrl = null;
  let feedbackTimer = null;

  function announce(text, visible = false) {
    feedback.textContent = text;
    feedback.dataset.visible = String(visible);
    clearTimeout(feedbackTimer);
    if (visible) {
      feedbackTimer = setTimeout(() => { feedback.dataset.visible = 'false'; }, 8000);
    }
  }

  function recording(active) {
    recordButton.setAttribute('aria-pressed', String(active));
    const label = active ? 'Stop recording' : 'Record screen';
    recordButton.setAttribute('aria-label', label);
    recordTooltip.textContent = label;
    recordButton.disabled = false;
  }

  function releaseStream() {
    if (stream) {
      // Remove ended handlers first, so stopping tracks cannot recursively stop.
      stream.getTracks().forEach(track => { track.onended = null; track.stop(); });
      stream = null;
    }
  }

  function stop() {
    if (recorder && recorder.state !== 'inactive') {
      recordButton.disabled = true;
      recorder.stop();
    }
  }

  function releaseDownload() {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = null;
    download.hidden = true;
    download.removeAttribute('href');
  }

  function save(chunks, type) {
    if (disposed) return;
    if (!chunks.length) {
      announce('No video was captured. Please try again.', true);
      return;
    }
    releaseDownload();
    const blob = new Blob(chunks, { type });
    downloadUrl = URL.createObjectURL(blob);
    const extension = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogv' : 'webm';
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    download.href = downloadUrl;
    download.download = `interview-assistant-${stamp}.${extension}`;
    // Keep a manual link available if the browser blocks the automatic download.
    download.hidden = false;
    download.click();
    announce('Recording ready. Download started.');
  }

  async function toggleRecording() {
    if (recorder && recorder.state !== 'inactive') {
      stop();
      return;
    }
    recordButton.disabled = true;
    releaseDownload();
    announce('Choose a tab, window or screen to record.');
    try {
      const capture = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      if (disposed) {
        capture.getTracks().forEach(track => track.stop());
        return;
      }
      stream = capture;
      const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
        .find(candidate => MediaRecorder.isTypeSupported(candidate));
      const currentRecorder = new MediaRecorder(stream, type ? { mimeType: type } : {});
      recorder = currentRecorder;
      const chunks = [];
      let failed = false;
      currentRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      currentRecorder.onerror = () => {
        failed = true;
        stop();
        releaseStream();
        if (!disposed) {
          announce('Recording failed. Please try again.', true);
        }
      };
      currentRecorder.onstop = () => {
        const mimeType = currentRecorder.mimeType || chunks[0]?.type || 'video/webm';
        releaseStream();
        if (!disposed) {
          recording(false);
          if (!failed) {
            try {
              save(chunks, mimeType);
            } catch {
              releaseDownload();
              announce('The video could not be saved. Please try recording again.', true);
            }
          }
        }
      };
      stream.getVideoTracks().forEach(track => { track.onended = stop; });
      currentRecorder.start(1000);
      recording(true);
      announce('Screen recording started. Use Stop recording to finish.');
    } catch (error) {
      releaseStream();
      recorder = null;
      if (!disposed) {
        recording(false);
        announce(error.name === 'NotAllowedError' || error.name === 'AbortError'
          ? 'Recording was not started.'
          : 'Screen recording is unavailable. Try a supported desktop browser.', true);
      }
    }
  }

  function printPage() { window.print(); }
  function dismissTooltips(event) {
    if (event.key === 'Escape') {
      controls.forEach(control => { control.dataset.tooltipDismissed = 'true'; });
    }
  }
  controls.forEach(control => {
    const reset = () => { delete control.dataset.tooltipDismissed; };
    for (const event of ['pointerenter', 'focusin']) {
      control.addEventListener(event, reset);
      tooltipListeners.push([control, event, reset]);
    }
  });
  parentElement.addEventListener('keydown', dismissTooltips);
  printButton.addEventListener('click', printPage);
  recordButton.addEventListener('click', toggleRecording);
  if (!navigator.mediaDevices?.getDisplayMedia || typeof MediaRecorder === 'undefined') {
    recordButton.disabled = true;
    recordTooltip.textContent = 'Screen recording is unavailable in this browser';
  }

  return () => {
    disposed = true;
    printButton.removeEventListener('click', printPage);
    recordButton.removeEventListener('click', toggleRecording);
    parentElement.removeEventListener('keydown', dismissTooltips);
    tooltipListeners.forEach(([control, event, listener]) => {
      control.removeEventListener(event, listener);
    });
    clearTimeout(feedbackTimer);
    stop();
    releaseStream();
    releaseDownload();
  };
}
