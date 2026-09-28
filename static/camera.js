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

  function attachToBox(box) {
    const video = box.querySelector('video');
    const canvas = box.querySelector('canvas');
    const thumbs = box.querySelector('[data-camera-thumbs]');
    const startButton = box.querySelector('[data-start-camera]');
    const captureButton = box.querySelector('[data-capture-frame]');
    const retakeButton = box.querySelector('[data-retake]');
    const hiddenFields = Array.from(box.querySelectorAll('[data-camera-field]'));
    const prompts = readPromptText(box);
    const maxSide = parseInt(box.dataset.maxSide || '960', 10);
    const quality = parseFloat(box.dataset.quality || '0.85');
    const form = box.closest('form');

    if (!video || !canvas || !startButton || !captureButton || hiddenFields.length === 0) {
      return;
    }

    const state = { stream: null, index: 0, started: false };

    function promptFor(index) {
      return prompts[index] || 'Capture the next photo';
    }

    function renderThumbs() {
      if (!thumbs) return;
      thumbs.textContent = '';
      hiddenFields.forEach(function (field) {
        if (!field.value) return;
        const img = document.createElement('img');
        img.src = field.value;
        img.alt = 'Captured photo';
        thumbs.appendChild(img);
      });
    }

    function refreshControls() {
      const done = state.index >= hiddenFields.length;
      captureButton.disabled = !state.started || done;
      if (retakeButton) retakeButton.hidden = state.index === 0;
      if (!state.started) return;
      if (done) {
        setStatus(box, hiddenFields.length > 1
          ? 'All photos captured. Check them below, then submit (or Retake).'
          : 'Photo captured. Check it below, then submit (or Retake).');
      } else {
        setStatus(box, promptFor(state.index));
      }
    }

    function captureCurrentFrame() {
      if (!state.stream || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
        setStatus(box, 'Camera is not ready yet. Please open the camera again.');
        return;
      }
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
      canvas.width = Math.round(sourceWidth * scale);
      canvas.height = Math.round(sourceHeight * scale);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      hiddenFields[state.index].value = canvas.toDataURL('image/jpeg', quality);
      state.index += 1;
      renderThumbs();
      refreshControls();
    }

    function retake() {
      hiddenFields.forEach(function (field) { field.value = ''; });
      state.index = 0;
      renderThumbs();
      refreshControls();
    }

    startButton.addEventListener('click', async function () {
      if (state.started && state.stream) {
        setStatus(box, 'Camera already active. ' + promptFor(state.index));
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus(box, 'This browser blocks the camera on this address. Open the site over HTTPS (or on localhost).');
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
          await new Promise(function (resolve) { video.addEventListener('loadeddata', resolve, { once: true }); });
        }
        state.started = true;
        refreshControls();
      } catch (error) {
        console.error('Camera access error', error);
        if (state.stream) state.stream.getTracks().forEach(function (track) { track.stop(); });
        state.stream = null;
        state.started = false;
        setStatus(box, 'Camera permission was denied or no camera is available.');
        refreshControls();
      }
    });

    captureButton.addEventListener('click', function () {
      if (!state.started || !state.stream) {
        setStatus(box, 'Open the camera before capturing your image.');
        return;
      }
      captureCurrentFrame();
    });

    if (retakeButton) retakeButton.addEventListener('click', retake);

    if (form) {
      form.addEventListener('submit', function (event) {
        const fileInput = form.querySelector('input[type=file]');
        const hasFile = fileInput && fileInput.files && fileInput.files.length > 0;
        const missing = hiddenFields.filter(function (field) { return !field.value; }).length;
        if (hasFile) return;
        if (missing > 0) {
          event.preventDefault();
          setStatus(box, missing === hiddenFields.length
            ? 'Please open the camera and capture ' + (hiddenFields.length > 1 ? 'all ' + hiddenFields.length + ' photos' : 'a photo') + ' first.'
            : 'Please capture the remaining ' + missing + ' photo(s) first. ' + promptFor(state.index));
          box.scrollIntoView({ behavior: 'smooth', block: 'center' });
          return;
        }
        if (state.stream) state.stream.getTracks().forEach(function (track) { track.stop(); });
      });
    }

    refreshControls();
    setStatus(box, 'Press "Open Camera" and allow camera access. ' + promptFor(0));
  }

  boxes.forEach(attachToBox);
})();
