(function () {
  const boxes = document.querySelectorAll('[data-camera-box]');

  function setStatus(box, message) {
    const node = box.querySelector('[data-camera-status]');
    if (node) node.textContent = message;
  }

  function readPromptText(box) {
    const prompts = box.dataset.prompts || '';
    return prompts.split('|').filter(Boolean);
  }

  function getStreamState(box) {
    return box.__cameraState || { stream: null, promptIndex: 0, started: false };
  }

  function updateButtonState(box, isReady) {
    const captureButton = box.querySelector('[data-capture-frame]');
    if (captureButton) {
      captureButton.disabled = !isReady;
    }
  }

  function attachToBox(box) {
    const video = box.querySelector('video');
    const canvas = box.querySelector('canvas');
    const preview = box.querySelector('[data-camera-preview]');
    const startButton = box.querySelector('[data-start-camera]');
    const captureButton = box.querySelector('[data-capture-frame]');
    const hiddenFields = Array.from(box.querySelectorAll('[data-camera-field]'));
    const prompts = readPromptText(box);

    if (!video || !canvas || !startButton || !captureButton || hiddenFields.length === 0) {
      return;
    }

    const state = getStreamState(box);
    box.__cameraState = state;

    function renderPreview(dataUrl) {
      if (preview) {
        preview.src = dataUrl;
        preview.hidden = false;
      }
    }

    function captureCurrentFrame() {
      if (!state.stream || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
        setStatus(box, 'Camera is not ready yet. Please open the camera again.');
        return;
      }

      // Some cameras report 4K+ dimensions. Keep captured form fields small
      // enough that three face samples fit comfortably under Flask's request
      // limit while retaining enough detail for face detection and OCR.
      const sourceWidth = video.videoWidth || 640;
      const sourceHeight = video.videoHeight || 480;
      const scale = Math.min(1, 1280 / Math.max(sourceWidth, sourceHeight));
      const width = Math.round(sourceWidth * scale);
      const height = Math.round(sourceHeight * scale);
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, width, height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.72);
      const field = hiddenFields[state.promptIndex % hiddenFields.length];
      if (field) {
        field.value = dataUrl;
      }

      renderPreview(dataUrl);

      const prompt = prompts[state.promptIndex] || 'Capture the next pose';
      const nextIndex = state.promptIndex + 1;
      state.promptIndex = nextIndex;

      if (hiddenFields.length > 1 && nextIndex < hiddenFields.length) {
        setStatus(box, `Next: ${prompt}`);
      } else {
        setStatus(box, 'All captures completed. Review the preview and submit the form.');
      }

      if (state.promptIndex >= hiddenFields.length) {
        captureButton.disabled = true;
      }
    }

    startButton.addEventListener('click', async function () {
      if (state.started && state.stream) {
        setStatus(box, 'Camera already active.');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });

        state.stream = stream;
        video.srcObject = stream;
        await video.play();
        if (video.readyState < 2) {
          await new Promise((resolve) => video.addEventListener('loadeddata', resolve, { once: true }));
        }
        state.started = true;
        setStatus(box, prompts[0] || 'Camera ready. Capture the first frame.');
        updateButtonState(box, true);
      } catch (error) {
        console.error('Camera access error', error);
        if (state.stream) {
          state.stream.getTracks().forEach((track) => track.stop());
        }
        state.stream = null;
        state.started = false;
        setStatus(box, 'Camera permission was denied or no camera is available.');
        updateButtonState(box, false);
      }
    });

    captureButton.addEventListener('click', function () {
      if (!state.started || !state.stream) {
        setStatus(box, 'Open the camera before capturing your image.');
        return;
      }
      captureCurrentFrame();
    });

    box.__captureCurrentFrame = captureCurrentFrame;
    updateButtonState(box, false);
    setStatus(box, 'Open the camera, keep one face visible, and follow each capture instruction.');
  }

  boxes.forEach(attachToBox);
})();
