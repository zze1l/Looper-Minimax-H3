import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Inject CSS stylesheet dynamically with resilient multi-path resolution
(function injectStyles() {
    const styleId = "looper-minimax-h3-styles";
    if (!document.getElementById(styleId)) {
        const link = document.createElement("link");
        link.id = styleId;
        link.rel = "stylesheet";
        link.type = "text/css";
        const cssPath = import.meta.url.includes("/js/") ? "../css/pre_looper_timeline.css" : "./pre_looper_timeline.css";
        link.href = new URL(cssPath, import.meta.url).href;
        document.head.appendChild(link);
    }
})();

app.registerExtension({
    name: "LooperMinimaxH3.PreLooper",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const supported = new Set([
            "prelooperminimax", "prelooper", "prelooper_minimax",
            "prelooperlatentpreviewminimax", "prelooper_latent_preview"
        ]);
        const lowerName = (nodeData.name || "").toLowerCase();
        if (!supported.has(lowerName)) return;

        const origNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = origNodeCreated ? origNodeCreated.apply(this, arguments) : undefined;
            setupPreLooperUI(this);
            return res;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            setupPreLooperUI(this);
            return res;
        };
    }
});

function setupPreLooperUI(node) {
    if (node.__looperUIInitialized) return;
    node.__looperUIInitialized = true;
    const videoWidget = node.widgets?.find(w => w.name === "video");
    const headWidget = node.widgets?.find(w => w.name === "head_a");
    const tailWidget = node.widgets?.find(w => w.name === "tail_b");
    const seamWidget = node.widgets?.find(w => w.name === "seam_frame");

    const promptWidget = node.widgets?.find(w => w.name === "prompt");

    if (headWidget && (headWidget.value === undefined || headWidget.value === null)) {
        headWidget.value = 0;
    }
    if (tailWidget && (tailWidget.value === undefined || tailWidget.value === null)) {
        tailWidget.value = 0;
    }
    if (seamWidget && (seamWidget.value === undefined || seamWidget.value === null)) {
        seamWidget.value = 0;
    }

    // 1. Compact prompt text box to 50% vertical height
    if (promptWidget) {
        promptWidget.computeSize = function (width) {
            return [width || 380, 56];
        };
        if (promptWidget.options) {
            promptWidget.options.minNodeSize = [300, 56];
        }
        const resizePromptEl = () => {
            const el = promptWidget.element || promptWidget.inputEl;
            if (el) {
                el.style.height = "56px";
                el.style.minHeight = "46px";
                el.style.maxHeight = "68px";
            }
        };
        resizePromptEl();
        setTimeout(resizePromptEl, 60);
        setTimeout(resizePromptEl, 250);
    }

    const state = {
        totalFrames: 120,
        fps: 24,
        duration: 5.0,
        width: 1280,
        height: 720,
        filename: videoWidget ? videoWidget.value : "",
        isDraggingHead: false,
        isDraggingTail: false,
        isDraggingBlock: false,
        isDraggingSeam: false,
        isUpdatingFromTimeline: false,
        dragStartX: 0,
        dragStartHead: 0,
        dragStartTail: 0,
        dragStartSeam: 0,
        dragActiveLength: 0
    };

    // Dedicated video file input supporting all video formats
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".mp4,.mov,.webm,.mkv,.avi,video/mp4,video/quicktime,video/webm,video/x-matroska,video/*";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    // Suppress any extra upload buttons & extra preview boxes created below the node
    const cleanupExtraWidgets = () => {
        node.imgs = undefined;
        if (!node.widgets) return;
        for (let i = node.widgets.length - 1; i >= 0; i--) {
            const w = node.widgets[i];
            // Remove extra button widgets so ONLY the panel bar button remains
            if (w.type === "button" && (
                w.name?.toLowerCase().includes("upload") || 
                w.name?.toLowerCase().includes("choose")
            )) {
                node.widgets.splice(i, 1);
            }
            // Remove duplicate preview widgets below the timeline
            else if (
                w.name === "video-preview" || 
                w.name === "$$canvas-image-preview" || 
                w.type === "IMAGE_PREVIEW" ||
                w.name === "preview"
            ) {
                if (typeof w.onRemove === "function") w.onRemove();
                node.widgets.splice(i, 1);
            }
        }
    };
    cleanupExtraWidgets();

    // 1. DOM Container for Timeline & Video Preview
    const container = document.createElement("div");
    container.className = "looper-widget-container";

    // 2. Upload Bar in DOM widget
    const uploadBar = document.createElement("div");
    uploadBar.className = "looper-upload-bar";

    const uploadBtn = document.createElement("button");
    uploadBtn.type = "button";
    uploadBtn.className = "looper-upload-btn";
    uploadBtn.innerHTML = `📁 Choose Video / Upload`;
    uploadBtn.addEventListener("click", () => fileInput.click());

    const uploadStatus = document.createElement("span");
    uploadStatus.className = "looper-upload-status";
    if (!videoWidget) {
        uploadBtn.style.display = "none";
        uploadStatus.innerHTML = "✨ <strong>Latent Preview</strong> (Decodificado por VAE)";
    } else {
        uploadStatus.textContent = "Drop video here or click button";
    }

    uploadBar.appendChild(uploadBtn);
    uploadBar.appendChild(uploadStatus);
    container.appendChild(uploadBar);

    // 3. VHS-Style Preprocess & Node Progress Bar
    const nodePbarContainer = document.createElement("div");
    nodePbarContainer.className = "looper-node-pbar-container";
    nodePbarContainer.innerHTML = `
        <div class="looper-node-pbar-header">
            <span id="lbl-node-pbar-title">⏱️ Preprocess & Latents</span>
            <span id="lbl-node-pbar-pct">Ready</span>
        </div>
        <div class="looper-node-pbar-track">
            <div class="looper-node-pbar-fill" id="node-pbar-fill"></div>
        </div>
    `;
    container.appendChild(nodePbarContainer);

    const pbarFill = nodePbarContainer.querySelector("#node-pbar-fill");
    const pbarTitle = nodePbarContainer.querySelector("#lbl-node-pbar-title");
    const pbarPct = nodePbarContainer.querySelector("#lbl-node-pbar-pct");

    // 4. Metadata Bar
    const metaBar = document.createElement("div");
    metaBar.className = "looper-meta-bar";
    metaBar.innerHTML = `
        <span class="looper-meta-tag">🎞️ <strong class="looper-meta-highlight" id="lbl-frames">--</strong>f</span>
        <span class="looper-meta-tag">⏱️ <span id="lbl-duration">--</span>s</span>
        <span class="looper-meta-tag">📐 <span id="lbl-res">--x--</span></span>
        <span class="looper-meta-tag">⚡ <span id="lbl-fps">24</span>fps</span>
        <span class="looper-meta-tag" id="tag-audio" title="Audio Track">🎵 <span id="lbl-audio">--</span></span>
        <span class="looper-meta-tag" title="Exact middle seam between Chunk A and Chunk B">⚡ Seam: <strong class="looper-meta-highlight" id="lbl-seam">f--</strong></span>
    `;
    container.appendChild(metaBar);

    const lblFrames = metaBar.querySelector("#lbl-frames");
    const lblDuration = metaBar.querySelector("#lbl-duration");
    const lblRes = metaBar.querySelector("#lbl-res");
    const lblFps = metaBar.querySelector("#lbl-fps");
    const lblAudio = metaBar.querySelector("#lbl-audio");
    const lblSeam = metaBar.querySelector("#lbl-seam");

    // 4. Video Player with Drag-and-Drop
    const videoWrapper = document.createElement("div");
    videoWrapper.className = "looper-video-wrapper";

    const videoEl = document.createElement("video");
    videoEl.className = "looper-video-preview";
    videoEl.controls = true;
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.autoplay = false;
    videoEl.loop = false;
    videoWrapper.appendChild(videoEl);

    const scrubBadge = document.createElement("div");
    scrubBadge.className = "looper-scrub-badge";
    scrubBadge.style.display = "none";
    videoWrapper.appendChild(scrubBadge);

    container.appendChild(videoWrapper);

    // 5. Timeline Graph with Segments and Handles
    const timelineWrapper = document.createElement("div");
    timelineWrapper.className = "looper-timeline-wrapper";

    const timelineLabels = document.createElement("div");
    timelineLabels.className = "looper-timeline-labels";
    timelineLabels.innerHTML = `
        <span id="lbl-head-info">Head: 0f</span>
        <span id="lbl-active-info">Active: 0f</span>
        <span id="lbl-tail-info">Tail: 0f</span>
    `;
    timelineWrapper.appendChild(timelineLabels);

    const lblHeadInfo = timelineLabels.querySelector("#lbl-head-info");
    const lblActiveInfo = timelineLabels.querySelector("#lbl-active-info");
    const lblTailInfo = timelineLabels.querySelector("#lbl-tail-info");

    const track = document.createElement("div");
    track.className = "looper-timeline-track";

    const segHead = document.createElement("div");
    segHead.className = "looper-segment looper-seg-head";
    segHead.textContent = "HEAD";
    track.appendChild(segHead);

    // Draggable active chunks block container
    const chunksBlock = document.createElement("div");
    chunksBlock.className = "looper-chunks-block";
    chunksBlock.title = "↔ Click and drag to slide the active chunks block (conserves length)";

    const dragHint = document.createElement("div");
    dragHint.className = "looper-chunks-drag-hint";
    dragHint.innerHTML = "↔ SLIDE ACTIVE BLOCK";
    chunksBlock.appendChild(dragHint);

    const segChunkA = document.createElement("div");
    segChunkA.className = "looper-segment looper-seg-chuncka";
    segChunkA.textContent = "CHUNK A";
    chunksBlock.appendChild(segChunkA);

    const guideA = document.createElement("div");
    guideA.className = "looper-guide-indicator";
    guideA.title = "MiniMax H3 Guide A (22f)";
    chunksBlock.appendChild(guideA);

    // Draggable middle seam marker (start of new video) with 22-frame temporal lock
    const seamMarker = document.createElement("div");
    seamMarker.className = "looper-seam-marker";
    seamMarker.title = "↔ Arrastra para mover el corte central / inicio de nuevo video (mínimo 22 frames bloqueados para cada chunk)";
    seamMarker.innerHTML = `
        <div class="looper-seam-handle-grip" title="Arrastra para mover corte central"></div>
        <div class="looper-seam-tag" title="Punto de inicio del nuevo video / Seam (Arrastra para ajustar)">⚡ SEAM</div>
    `;
    chunksBlock.appendChild(seamMarker);

    const segChunkB = document.createElement("div");
    segChunkB.className = "looper-segment looper-seg-chunckb";
    segChunkB.textContent = "CHUNK B";
    chunksBlock.appendChild(segChunkB);

    const guideB = document.createElement("div");
    guideB.className = "looper-guide-indicator";
    guideB.title = "MiniMax H3 Guide B (22f)";
    chunksBlock.appendChild(guideB);

    track.appendChild(chunksBlock);

    const segTail = document.createElement("div");
    segTail.className = "looper-segment looper-seg-tail";
    segTail.textContent = "TAIL";
    track.appendChild(segTail);

    const handleHead = document.createElement("div");
    handleHead.className = "looper-handle";
    handleHead.title = "Drag to adjust head_a cut";
    handleHead.innerHTML = `<div class="looper-handle-grip"></div>`;
    track.appendChild(handleHead);

    const handleTail = document.createElement("div");
    handleTail.className = "looper-handle";
    handleTail.title = "Drag to adjust tail_b cut";
    handleTail.innerHTML = `<div class="looper-handle-grip"></div>`;
    track.appendChild(handleTail);

    const playhead = document.createElement("div");
    playhead.className = "looper-playhead-indicator";
    playhead.style.display = "none";
    track.appendChild(playhead);

    timelineWrapper.appendChild(track);

    const legend = document.createElement("div");
    legend.className = "looper-legend";
    legend.innerHTML = `
        <span class="looper-legend-item"><span class="looper-dot looper-dot-cut"></span> Cut</span>
        <span class="looper-legend-item"><span class="looper-dot looper-dot-chuncka"></span> Chunk A</span>
        <span class="looper-legend-item"><span class="looper-dot looper-dot-seam"></span> Seam (Mid)</span>
        <span class="looper-legend-item"><span class="looper-dot looper-dot-chunckb"></span> Chunk B</span>
        <span class="looper-legend-item"><span class="looper-dot looper-dot-guide"></span> 22f Guide</span>
    `;
    timelineWrapper.appendChild(legend);
    container.appendChild(timelineWrapper);

    // 6. Video Upload Function
    async function uploadVideoFile(file) {
        if (!file) return;
        uploadStatus.textContent = `⏳ Uploading ${file.name}...`;
        uploadBtn.innerHTML = `⏳ Uploading...`;
        uploadBtn.disabled = true;
        node.setDirtyCanvas(true, true);

        const formData = new FormData();
        formData.append("video", file);
        formData.append("file", file);

        try {
            let uploadedName = "";
            let meta = null;

            let resp = await api.fetchApi("/looper_h3/upload_video", {
                method: "POST",
                body: formData
            });

            if (resp.ok) {
                const data = await resp.json();
                uploadedName = data.name || data.filename;
                meta = data.metadata;
            } else {
                const fbFormData = new FormData();
                fbFormData.append("image", file);
                fbFormData.append("overwrite", "true");
                const fbResp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body: fbFormData
                });
                if (fbResp.ok) {
                    const fbData = await fbResp.json();
                    uploadedName = fbData.name;
                }
            }

            if (uploadedName) {
                uploadStatus.textContent = `✅ ${uploadedName}`;

                if (videoWidget) {
                    if (videoWidget.options && Array.isArray(videoWidget.options.values)) {
                        if (!videoWidget.options.values.includes(uploadedName)) {
                            videoWidget.options.values.push(uploadedName);
                            videoWidget.options.values.sort();
                        }
                    }
                    videoWidget.value = uploadedName;
                    if (videoWidget.callback) videoWidget.callback(uploadedName);
                }

                if (meta && meta.total_frames) {
                    state.totalFrames = meta.total_frames;
                    state.fps = meta.fps || 24;
                    state.duration = meta.duration || (meta.total_frames / state.fps);
                    state.width = meta.width || 1280;
                    state.height = meta.height || 720;
                }

                loadVideo(uploadedName);
                cleanupExtraWidgets();
                node.setDirtyCanvas(true, true);
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            } else {
                uploadStatus.textContent = `❌ Upload failed`;
                node.setDirtyCanvas(true, true);
            }
        } catch (err) {
            console.error("[LooperMinimaxH3] Upload error:", err);
            uploadStatus.textContent = `❌ Error: ${err.message}`;
            node.setDirtyCanvas(true, true);
        } finally {
            uploadBtn.innerHTML = `📁 Choose Video / Upload`;
            uploadBtn.disabled = false;
            fileInput.value = "";
        }
    }

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            uploadVideoFile(fileInput.files[0]);
        }
    });

    // Drag & Drop on video wrapper
    videoWrapper.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        videoWrapper.classList.add("dragover");
    });

    videoWrapper.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        videoWrapper.classList.remove("dragover");
    });

    videoWrapper.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        videoWrapper.classList.remove("dragover");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadVideoFile(e.dataTransfer.files[0]);
        }
    });

    // 7. Badge Notification Helper
    let badgeTimer = null;
    function showBadge(html, autoHideMs = 0) {
        if (badgeTimer) {
            clearTimeout(badgeTimer);
            badgeTimer = null;
        }
        scrubBadge.style.display = "block";
        scrubBadge.innerHTML = html;
        if (autoHideMs > 0) {
            badgeTimer = setTimeout(() => {
                scrubBadge.style.display = "none";
            }, autoHideMs);
        }
    }

    // Update Timeline Graphics & Visual Preview (head cut, tail cut, or movable seam cut)
    function updateTimelineDisplay(seekTarget = false) {
        const total = Math.max(1, state.totalFrames);
        let headVal = headWidget ? Number(headWidget.value) || 0 : 0;
        let tailVal = tailWidget ? Number(tailWidget.value) || 0 : 0;
        let seamVal = seamWidget ? Number(seamWidget.value) || 0 : 0;

        headVal = Math.max(0, Math.min(headVal, total - 44));
        tailVal = Math.max(0, Math.min(tailVal, total - headVal - 44));

        const activeFrames = Math.max(0, total - headVal - tailVal);
        const minSeam = headVal + 22;
        const maxSeam = Math.max(minSeam, (total - tailVal) - 22);

        // Seam frame (where the new looped video begins)
        let seamFrame;
        if (seamVal > 0) {
            seamFrame = Math.max(minSeam, Math.min(seamVal, maxSeam));
        } else {
            // Default: auto 50/50 center split
            const halfActive = Math.floor(activeFrames / 2);
            seamFrame = Math.max(minSeam, Math.min(headVal + halfActive, maxSeam));
        }

        const fps = state.fps || 24;
        const seamTime = seamFrame / fps;

        // Cut point frames & times
        const headFrame = Math.max(0, Math.min(total - 1, headVal));
        const headTime = headFrame / fps;

        const tailCutFrame = Math.max(0, Math.min(total - 1, total - tailVal));
        const tailTime = tailCutFrame / fps;

        const headPct = (headVal / total) * 100;
        const tailPct = (tailVal / total) * 100;
        const activePct = (activeFrames / total) * 100;

        segHead.style.left = "0%";
        segHead.style.width = `${headPct}%`;
        segHead.style.display = headVal > 0 ? "flex" : "none";

        handleHead.style.left = `${headPct}%`;

        // Position Chunks Block spanning active window
        chunksBlock.style.left = `${headPct}%`;
        chunksBlock.style.width = `${activePct}%`;

        // Chunk A and Chunk B frame counts respecting 22 frames minimum constraint
        const chunkAFrames = Math.max(22, seamFrame - headVal);
        const chunkBFrames = Math.max(22, (total - tailVal) - seamFrame);

        const chunkAPctOfActive = activeFrames > 0 ? (chunkAFrames / activeFrames) * 100 : 50;
        const chunkBPctOfActive = 100 - chunkAPctOfActive;

        segChunkA.style.left = "0%";
        segChunkA.style.width = `${chunkAPctOfActive}%`;

        // 22f Guide A: locked to the beginning of Chunk A
        const guideAPctOfActive = activeFrames > 0 ? Math.min(chunkAPctOfActive, (22 / activeFrames) * 100) : 0;
        guideA.style.left = "0%";
        guideA.style.width = `${guideAPctOfActive}%`;

        // Movable Seam Marker right at junction (new video start)
        seamMarker.style.left = `${chunkAPctOfActive}%`;
        const seamTag = seamMarker.querySelector(".looper-seam-tag");
        if (seamTag) {
            seamTag.textContent = `⚡ SEAM f${seamFrame}`;
        }

        segChunkB.style.left = `${chunkAPctOfActive}%`;
        segChunkB.style.width = `${chunkBPctOfActive}%`;

        // 22f Guide B: locked to the end of Chunk B
        const guideBPctOfActive = activeFrames > 0 ? Math.min(chunkBPctOfActive, (22 / activeFrames) * 100) : 0;
        guideB.style.left = `${100 - guideBPctOfActive}%`;
        guideB.style.width = `${guideBPctOfActive}%`;

        segTail.style.left = `${100 - tailPct}%`;
        segTail.style.width = `${tailPct}%`;
        segTail.style.display = tailVal > 0 ? "flex" : "none";

        handleTail.style.left = `${100 - tailPct}%`;

        lblHeadInfo.innerHTML = `Head: <strong>${headVal}f</strong> (${headPct.toFixed(1)}%)`;
        lblActiveInfo.innerHTML = `Active: <strong>${activeFrames}f</strong> (Chunk A: <strong>${chunkAFrames}f</strong> | ⚡ Seam / Start: <strong>f${seamFrame}</strong> | Chunk B: <strong>${chunkBFrames}f</strong>)`;
        lblTailInfo.innerHTML = `Tail: <strong>${tailVal}f</strong> (${tailPct.toFixed(1)}%)`;

        if (lblSeam) {
            lblSeam.textContent = `f${seamFrame} (${seamTime.toFixed(2)}s)`;
        }

        // Preview target: head cut point, tail cut point, or mid seam
        if (seekTarget === true || seekTarget === "mid" || seekTarget === "seam") {
            videoEl.pause();
            videoEl.currentTime = seamTime;
        } else if (seekTarget === "head") {
            videoEl.pause();
            videoEl.currentTime = headTime;
        } else if (seekTarget === "tail") {
            videoEl.pause();
            videoEl.currentTime = tailTime;
        } else if (typeof seekTarget === "number") {
            videoEl.pause();
            videoEl.currentTime = seekTarget;
        }

        return { 
            headVal, 
            tailVal, 
            activeFrames, 
            seamFrame, 
            seamTime, 
            chunkAFrames, 
            chunkBFrames, 
            headFrame, 
            headTime, 
            tailCutFrame, 
            tailTime 
        };
    }

    // 8. Video Loading
    async function loadVideo(filename, type = "input") {
        if (!filename) return;
        state.filename = filename;
        uploadStatus.textContent = filename;

        const videoPath = `/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(type)}`;
        const videoUrl = typeof api.apiURL === "function" ? api.apiURL(videoPath) : videoPath;
        loadVideoUrl(videoUrl, filename);

        if (type === "input") {
            try {
                const resp = await api.fetchApi(`/looper_h3/video_info?filename=${encodeURIComponent(filename)}`);
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.total_frames > 0) {
                        state.totalFrames = data.total_frames;
                        state.fps = data.fps || 24;
                        state.duration = data.duration || (data.total_frames / state.fps);
                        state.width = data.width || 1280;
                        state.height = data.height || 720;
                        state.hasAudio = Boolean(data.has_audio);
                        renderMetadata();
                        updateTimelineDisplay("mid");
                    }
                }
            } catch (e) {
                console.warn("[LooperMinimaxH3] Using HTML5 video metadata:", e);
            }
        }
    }

    function loadVideoUrl(videoUrl, filename) {
        if (!videoUrl) return;
        state.filename = filename || "video_preview.mp4";
        uploadStatus.textContent = `🎬 ${state.filename}`;
        videoEl.src = videoUrl;
        try {
            videoEl.load();
        } catch (_) {}

        videoEl.onloadedmetadata = () => {
            if (videoEl.duration) {
                state.duration = videoEl.duration;
                state.totalFrames = Math.max(1, Math.round(videoEl.duration * (state.fps || 24)));
            }
            if (videoEl.videoWidth) {
                state.width = videoEl.videoWidth;
                state.height = videoEl.videoHeight;
            }
            renderMetadata();
            updateTimelineDisplay("mid");
            node.setDirtyCanvas(true, true);
        };

        renderMetadata();
        updateTimelineDisplay();
    }

    function renderMetadata() {
        lblFrames.textContent = state.totalFrames;
        lblDuration.textContent = state.duration.toFixed(1);
        lblRes.textContent = `${state.width}x${state.height}`;
        lblFps.textContent = state.fps;
        if (lblAudio) {
            lblAudio.textContent = state.hasAudio ? "Audio" : "No Audio";
            lblAudio.style.color = state.hasAudio ? "#34d399" : "#94a3b8";
        }
    }

    // 8b. Active Window Loop Preview Player
    // Plays smoothly within the active range [headTime, endTime], looping seamlessly
    videoEl.addEventListener("play", () => {
        if (state.isDraggingHead || state.isDraggingTail || state.isDraggingBlock || state.isDraggingSeam) return;
        const total = Math.max(1, state.totalFrames);
        const fps = state.fps || 24;
        let headVal = headWidget ? Number(headWidget.value) || 0 : 0;
        let tailVal = tailWidget ? Number(tailWidget.value) || 0 : 0;
        const endFrame = Math.max(0, total - tailVal);
        const endTime = endFrame / fps;
        const headTime = headVal / fps;

        // If play starts outside active range or near the end, loop to start of active range
        if (videoEl.currentTime < headTime || videoEl.currentTime >= endTime - 0.05) {
            videoEl.currentTime = headTime;
        }
        playhead.style.display = "block";
    });

    videoEl.addEventListener("pause", () => {
        if (!state.isDraggingHead && !state.isDraggingTail && !state.isDraggingBlock && !state.isDraggingSeam) {
            playhead.style.display = "none";
        }
    });

    videoEl.addEventListener("ended", () => {
        const total = Math.max(1, state.totalFrames);
        const fps = state.fps || 24;
        let headVal = headWidget ? Number(headWidget.value) || 0 : 0;
        videoEl.currentTime = headVal / fps;
        videoEl.play().catch(() => {});
    });

    videoEl.addEventListener("timeupdate", () => {
        if (videoEl.paused) return;
        if (state.isDraggingHead || state.isDraggingTail || state.isDraggingBlock || state.isDraggingSeam) return;

        const total = Math.max(1, state.totalFrames);
        const fps = state.fps || 24;
        let headVal = headWidget ? Number(headWidget.value) || 0 : 0;
        let tailVal = tailWidget ? Number(tailWidget.value) || 0 : 0;
        const endFrame = Math.max(0, total - tailVal);
        const endTime = endFrame / fps;
        const headTime = headVal / fps;

        // Update live playhead position on timeline track
        const currentFrame = Math.round(videoEl.currentTime * fps);
        const pct = Math.max(0, Math.min(100, (currentFrame / total) * 100));
        playhead.style.display = "block";
        playhead.style.left = `${pct}%`;

        // Seamless loop when reaching tail cut point
        if (videoEl.currentTime >= endTime - 0.05) {
            videoEl.currentTime = headTime;
        }
    });

    // 9. Interactive Dragging: Visualizing Cut Points for Head & Tail, Movable Seam with 22f Constraint
    function onMouseMove(e) {
        if (!state.isDraggingHead && !state.isDraggingTail && !state.isDraggingBlock && !state.isDraggingSeam) return;

        const rect = track.getBoundingClientRect();
        const clientX = e.clientX;
        const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const total = Math.max(1, state.totalFrames);
        const targetFrame = Math.round(ratio * total);

        videoEl.pause();

        if (state.isDraggingSeam) {
            // Dragging middle cut / seam: move where new video starts, blocking 22f minimum on each side
            const headVal = headWidget ? Number(headWidget.value) || 0 : 0;
            const tailVal = tailWidget ? Number(tailWidget.value) || 0 : 0;
            const minSeam = headVal + 22;
            const maxSeam = Math.max(minSeam, (total - tailVal) - 22);
            const newSeam = Math.max(minSeam, Math.min(targetFrame, maxSeam));

            state.isUpdatingFromTimeline = true;
            if (seamWidget) {
                seamWidget.value = newSeam;
                if (seamWidget.callback) seamWidget.callback(newSeam);
            }
            state.isUpdatingFromTimeline = false;

            const { seamFrame, seamTime, chunkAFrames, chunkBFrames } = updateTimelineDisplay("mid");
            showBadge(`⚡ Inicio Nuevo Video: Frame <strong>${seamFrame}</strong> (${seamTime.toFixed(2)}s) | Chunk A: ${chunkAFrames}f | Chunk B: ${chunkBFrames}f [22f bloqueados]`);
        } else if (state.isDraggingBlock) {
            const deltaX = e.clientX - state.dragStartX;
            const deltaFrames = Math.round((deltaX / rect.width) * total);

            const activeLen = state.dragActiveLength;
            const maxHead = Math.max(0, total - activeLen);
            let newHead = state.dragStartHead + deltaFrames;
            newHead = Math.max(0, Math.min(newHead, maxHead));
            let newTail = total - newHead - activeLen;
            newTail = Math.max(0, newTail);

            // Shift seam synchronously with active block
            const shiftHead = newHead - state.dragStartHead;
            let newSeam = state.dragStartSeam + shiftHead;
            const minSeam = newHead + 22;
            const maxSeam = Math.max(minSeam, (total - newTail) - 22);
            newSeam = Math.max(minSeam, Math.min(newSeam, maxSeam));

            state.isUpdatingFromTimeline = true;
            if (headWidget) {
                headWidget.value = newHead;
                if (headWidget.callback) headWidget.callback(newHead);
            }
            if (tailWidget) {
                tailWidget.value = newTail;
                if (tailWidget.callback) tailWidget.callback(newTail);
            }
            if (seamWidget) {
                seamWidget.value = newSeam;
                if (seamWidget.callback) seamWidget.callback(newSeam);
            }
            state.isUpdatingFromTimeline = false;

            const { seamFrame, seamTime, chunkAFrames, chunkBFrames } = updateTimelineDisplay("mid");
            showBadge(`↔ Desplazando Bloque: <strong>${activeLen}f</strong> | Head: ${newHead}f | Tail: ${newTail}f | ⚡ Inicio: <strong>f${seamFrame}</strong> (A: ${chunkAFrames}f | B: ${chunkBFrames}f)`);
        } else if (state.isDraggingHead) {
            const tailVal = tailWidget ? Number(tailWidget.value) || 0 : 0;
            const maxHead = Math.max(0, total - tailVal - 44);
            const newHead = Math.max(0, Math.min(targetFrame, maxHead));

            state.isUpdatingFromTimeline = true;
            if (headWidget) {
                headWidget.value = newHead;
                if (headWidget.callback) headWidget.callback(newHead);
            }
            // Enforce seam doesn't violate 22 frames minimum for Chunk A
            if (seamWidget && Number(seamWidget.value) > 0) {
                const minSeam = newHead + 22;
                if (Number(seamWidget.value) < minSeam) {
                    seamWidget.value = minSeam;
                    if (seamWidget.callback) seamWidget.callback(minSeam);
                }
            }
            state.isUpdatingFromTimeline = false;

            // Visualizing the exact point cut at the Head
            const { headFrame, headTime, seamFrame } = updateTimelineDisplay("head");
            showBadge(`✂️ Corte Head (Inicio): Frame <strong>${headFrame}</strong> / ${total} (${headTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`);
        } else if (state.isDraggingTail) {
            const headVal = headWidget ? Number(headWidget.value) || 0 : 0;
            const maxTail = Math.max(0, total - headVal - 44);
            const newTail = Math.max(0, Math.min(total - targetFrame, maxTail));

            state.isUpdatingFromTimeline = true;
            if (tailWidget) {
                tailWidget.value = newTail;
                if (tailWidget.callback) tailWidget.callback(newTail);
            }
            // Enforce seam doesn't violate 22 frames minimum for Chunk B
            if (seamWidget && Number(seamWidget.value) > 0) {
                const maxSeam = (total - newTail) - 22;
                if (Number(seamWidget.value) > maxSeam) {
                    seamWidget.value = maxSeam;
                    if (seamWidget.callback) seamWidget.callback(maxSeam);
                }
            }
            state.isUpdatingFromTimeline = false;

            // Visualizing the exact point cut at the Tail
            const { tailCutFrame, tailTime, seamFrame } = updateTimelineDisplay("tail");
            showBadge(`✂️ Corte Tail (Final): Frame <strong>${tailCutFrame}</strong> (-${newTail}f, ${tailTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`);
        }

        node.setDirtyCanvas(true, true);
    }

    function onMouseUp() {
        const wasActive = state.isDraggingHead || state.isDraggingTail || state.isDraggingBlock || state.isDraggingSeam;
        state.isDraggingHead = false;
        state.isDraggingTail = false;
        state.isDraggingBlock = false;
        state.isDraggingSeam = false;
        handleHead.classList.remove("dragging");
        handleTail.classList.remove("dragging");
        chunksBlock.classList.remove("dragging");
        seamMarker.classList.remove("dragging");

        // Keep preview paused on cut/seam frame and hide badge smoothly
        if (wasActive) {
            if (badgeTimer) clearTimeout(badgeTimer);
            badgeTimer = setTimeout(() => {
                scrubBadge.style.display = "none";
            }, 1400);
        } else {
            scrubBadge.style.display = "none";
        }

        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
    }

    // Drag handle head cut: preview head cut frame immediately on click
    handleHead.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        state.isDraggingHead = true;
        handleHead.classList.add("dragging");
        const { headFrame, headTime, seamFrame } = updateTimelineDisplay("head");
        showBadge(`✂️ Corte Head: Frame <strong>${headFrame}</strong> (${headTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`);
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
    });

    // Drag handle tail cut: preview tail cut frame immediately on click
    handleTail.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        state.isDraggingTail = true;
        handleTail.classList.add("dragging");
        const { tailCutFrame, tailTime, seamFrame } = updateTimelineDisplay("tail");
        showBadge(`✂️ Corte Tail: Frame <strong>${tailCutFrame}</strong> (${tailTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`);
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
    });

    // Drag middle seam marker directly (where new video starts)
    seamMarker.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        state.isDraggingSeam = true;
        seamMarker.classList.add("dragging");
        const { seamFrame, seamTime, chunkAFrames, chunkBFrames } = updateTimelineDisplay("mid");
        showBadge(`⚡ Inicio Nuevo Video: Frame <strong>${seamFrame}</strong> (${seamTime.toFixed(2)}s) | Chunk A: ${chunkAFrames}f | Chunk B: ${chunkBFrames}f [22f bloqueados]`);
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
    });

    // Drag entire active chunks block horizontally
    chunksBlock.addEventListener("mousedown", (e) => {
        // If clicked on seam marker or handles, let them handle dragging
        if (e.target.closest(".looper-seam-marker") || e.target.closest(".looper-handle")) return;

        e.preventDefault();
        e.stopPropagation();

        state.isDraggingBlock = true;
        state.dragStartX = e.clientX;
        state.dragStartHead = headWidget ? Number(headWidget.value) || 0 : 0;
        state.dragStartTail = tailWidget ? Number(tailWidget.value) || 0 : 0;
        const total = Math.max(1, state.totalFrames);
        state.dragActiveLength = Math.max(44, total - state.dragStartHead - state.dragStartTail);
        state.dragStartSeam = seamWidget && Number(seamWidget.value) > 0 ? Number(seamWidget.value) : (state.dragStartHead + Math.floor(state.dragActiveLength / 2));

        chunksBlock.classList.add("dragging");
        const { seamFrame, seamTime } = updateTimelineDisplay("mid");
        showBadge(`↔ Desplazar Bloque: <strong>${state.dragActiveLength}f</strong> | ⚡ Seam: f${seamFrame} (${seamTime.toFixed(2)}s)`);

        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
    });

    // Click on Seam in metadata bar to preview seam
    if (lblSeam) {
        lblSeam.style.cursor = "pointer";
        lblSeam.title = "Click para visualizar el corte central (seam) de Chunk A y Chunk B";
        lblSeam.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const { seamFrame, seamTime } = updateTimelineDisplay("mid");
            showBadge(`⚡ Corte Central (Seam): Frame <strong>${seamFrame}</strong> (${seamTime.toFixed(2)}s)`, 1800);
        });
    }

    // 10. Automated Cut Point & Seam Preview when moving widget inputs/sliders directly
    if (headWidget) {
        const origHeadCb = headWidget.callback;
        headWidget.callback = function (val) {
            if (origHeadCb) origHeadCb.apply(this, arguments);
            if (!state.isUpdatingFromTimeline) {
                const { headFrame, headTime, seamFrame } = updateTimelineDisplay("head");
                showBadge(`✂️ Corte Head: Frame <strong>${headFrame}</strong> (${headTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`, 1800);
            }
        };
    }

    if (tailWidget) {
        const origTailCb = tailWidget.callback;
        tailWidget.callback = function (val) {
            if (origTailCb) origTailCb.apply(this, arguments);
            if (!state.isUpdatingFromTimeline) {
                const { tailCutFrame, tailTime, seamFrame } = updateTimelineDisplay("tail");
                showBadge(`✂️ Corte Tail: Frame <strong>${tailCutFrame}</strong> (-${val}f, ${tailTime.toFixed(2)}s) | ⚡ Seam: f${seamFrame}`, 1800);
            }
        };
    }

    if (seamWidget) {
        const origSeamCb = seamWidget.callback;
        seamWidget.callback = function (val) {
            if (origSeamCb) origSeamCb.apply(this, arguments);
            if (!state.isUpdatingFromTimeline) {
                const { seamFrame, seamTime, chunkAFrames, chunkBFrames } = updateTimelineDisplay("mid");
                showBadge(`⚡ Inicio Nuevo Video: Frame <strong>${seamFrame}</strong> (${seamTime.toFixed(2)}s) | A: ${chunkAFrames}f | B: ${chunkBFrames}f [22f min]`, 1800);
            }
        };
    }

    if (videoWidget) {
        const origVideoCb = videoWidget.callback;
        videoWidget.callback = function (val) {
            if (origVideoCb) origVideoCb.apply(this, arguments);
            loadVideo(val);
            cleanupExtraWidgets();
        };
        if (videoWidget.value) {
            loadVideo(videoWidget.value);
        }
    }

    // 11. ComfyUI API execution & progress listeners for live node preprocessing metrics
    let preprocessStartTime = 0;
    const onProgress = ({ detail }) => {
        if (detail && detail.node === node.id) {
            const pct = Math.round((detail.value / detail.max) * 100);
            if (pbarFill) {
                pbarFill.style.width = `${pct}%`;
                pbarFill.classList.add("active");
            }
            if (pbarPct) pbarPct.textContent = `${pct}%`;
            if (pbarTitle) pbarTitle.textContent = `⏳ Preprocessing (${pct}%)...`;
        }
    };

    const onExecuting = ({ detail }) => {
        if (detail === node.id) {
            preprocessStartTime = performance.now();
            if (pbarFill) {
                pbarFill.style.width = "5%";
                pbarFill.classList.add("active");
            }
            if (pbarTitle) pbarTitle.textContent = `⏳ Preprocessing & Latents...`;
            if (pbarPct) pbarPct.textContent = `Running...`;
        } else if (detail === null && preprocessStartTime > 0) {
            const elapsed = ((performance.now() - preprocessStartTime) / 1000).toFixed(2);
            preprocessStartTime = 0;
            if (pbarFill) {
                pbarFill.style.width = "100%";
                pbarFill.classList.remove("active");
            }
            if (pbarTitle) pbarTitle.textContent = `✅ Preprocess Finished`;
            if (pbarPct) pbarPct.textContent = `${elapsed}s`;
        }
    };

    api.addEventListener("progress", onProgress);
    api.addEventListener("executing", onExecuting);

    function handleExecutedOutput(output) {
        if (!output) return;
        const videos = output?.videos || output?.video || output?.gifs || output?.images;
        if (Array.isArray(videos) && videos.length > 0) {
            const v = videos[0];
            const fname = v.filename || v.name;
            const sub = v.subfolder || "";
            const vtype = v.type || "temp";
            if (fname) {
                const videoPath = `/view?filename=${encodeURIComponent(fname)}&type=${encodeURIComponent(vtype)}&subfolder=${encodeURIComponent(sub)}`;
                const videoUrl = typeof api.apiURL === "function" ? api.apiURL(videoPath) : videoPath;
                loadVideoUrl(videoUrl, fname);
            }
        }
    }

    const onExecuted = (message) => {
        const detail = message?.detail || message;
        if (!detail) return;
        if (detail.node && String(detail.node) !== String(node.id)) return;

        const output = detail.output || detail;
        handleExecutedOutput(output);
    };
    api.addEventListener("executed", onExecuted);

    const origOnExecuted = node.onExecuted;
    node.onExecuted = function(message) {
        if (origOnExecuted) origOnExecuted.apply(this, arguments);
        handleExecutedOutput(message);
    };

    // Check cached output or history on initialization & configure
    async function checkExistingVideoOutput() {
        if (state.filename && videoEl.src && videoEl.src.length > 5 && !videoEl.src.endsWith("/")) return;
        
        // 1. Check in-memory app.nodeOutputs
        if (app.nodeOutputs && app.nodeOutputs[String(node.id)]) {
            handleExecutedOutput(app.nodeOutputs[String(node.id)]);
            return;
        }

        // 2. Check /history API
        try {
            const resp = await api.fetchApi("/history");
            if (resp.ok) {
                const history = await resp.json();
                const pids = Object.keys(history).reverse();
                for (const pid of pids) {
                    const item = history[pid];
                    const out = item?.outputs?.[String(node.id)];
                    if (out) {
                        handleExecutedOutput(out);
                        return;
                    }
                }
            }
        } catch (e) {
            console.debug("[LooperMinimaxH3] History fetch skipped:", e);
        }
    }

    const onExecutionCached = ({ detail }) => {
        if (detail?.nodes && detail.nodes.map(String).includes(String(node.id))) {
            setTimeout(checkExistingVideoOutput, 50);
        }
    };
    api.addEventListener("execution_cached", onExecutionCached);

    setTimeout(checkExistingVideoOutput, 150);
    setTimeout(checkExistingVideoOutput, 600);

    // 12. External Load Video connection detection & priority sync
    function getExternalVideoConnection() {
        if (!node.inputs || !app.graph) return null;
        const input = node.inputs.find(inp => inp.name === "load_video" || inp.name === "video_images" || inp.name === "preview_video" || inp.name === "latent");
        if (!input || input.link == null) return null;
        const link = app.graph.links ? app.graph.links[input.link] : null;
        if (!link) return null;
        const originNode = app.graph.getNodeById ? app.graph.getNodeById(link.origin_id) : null;
        return { input, link, originNode };
    }

    let lastConnId = -1;
    function syncExternalConnectionUI() {
        const conn = getExternalVideoConnection();
        const currentConnId = conn ? conn.link.id : -1;
        if (conn) {
            const originTitle = conn.originNode?.title || conn.originNode?.type || "Load Video";
            uploadStatus.innerHTML = `🔗 <strong style="color:#38bdf8;">${originTitle}</strong> <span style="font-size:10px; font-weight:bold; color:#10b981; background:rgba(16,185,129,0.15); border:1px solid #10b981; padding:1px 6px; border-radius:4px; margin-left:4px;">PRIORIDAD ACTIVA</span>`;
            uploadBtn.style.opacity = "0.65";
            uploadBtn.title = `Nodo '${originTitle}' enlazado. Tiene prioridad activa sobre la carga local por botón/selector.`;

            // If the connected origin node (e.g. VHS_LoadVideo) has a video filename, preview it!
            if (conn.originNode && conn.originNode.widgets) {
                const originVidWidget = conn.originNode.widgets.find(w => 
                    w.name === "video" || w.name === "video_path" || w.name === "file" || w.name === "choose_video"
                );
                if (originVidWidget) {
                    if (!originVidWidget.__looperHooked) {
                        originVidWidget.__looperHooked = true;
                        const origCb = originVidWidget.callback;
                        originVidWidget.callback = function (val) {
                            if (origCb) origCb.apply(this, arguments);
                            if (getExternalVideoConnection()) {
                                loadVideo(val);
                            }
                        };
                    }
                    if (originVidWidget.value && originVidWidget.value !== state.filename && currentConnId !== lastConnId) {
                        loadVideo(originVidWidget.value);
                    }
                }
            }
        } else if (lastConnId !== -1) {
            uploadBtn.style.opacity = "1.0";
            uploadBtn.title = "";
            if (videoWidget && videoWidget.value && videoWidget.value !== "none.mp4") {
                uploadStatus.textContent = videoWidget.value;
                loadVideo(videoWidget.value);
            } else {
                uploadStatus.textContent = videoWidget ? "Drop video here or click button" : "Latent VAE preview (run node to view)";
            }
        }
        lastConnId = currentConnId;
    }

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, isConnected, link_info, input_or_output) {
        const res = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
        syncExternalConnectionUI();
        node.setDirtyCanvas(true, true);
        return res;
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const res = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
        setTimeout(syncExternalConnectionUI, 120);
        setTimeout(checkExistingVideoOutput, 180);
        return res;
    };

    // 13. Lifecycle handling & cleanup
    const origDrawBg = node.onDrawBackground;
    node.onDrawBackground = function(ctx) {
        node.imgs = undefined;
        cleanupExtraWidgets();
        syncExternalConnectionUI();
        if (origDrawBg) origDrawBg.apply(this, arguments);
    };

    const origOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (fileInput && fileInput.parentNode) {
            fileInput.remove();
        }
        api.removeEventListener("progress", onProgress);
        api.removeEventListener("executing", onExecuting);
        api.removeEventListener("executed", onExecuted);
        api.removeEventListener("execution_cached", onExecutionCached);
        if (origOnRemoved) origOnRemoved.apply(this, arguments);
    };

    // 14. Add DOM Widget to Node
    node.addDOMWidget("pre_looper_timeline_ui", "timeline_widget", container, {
        serialize: false,
        hideOnZoom: false
    });

    node.size = [Math.max(node.size[0], 420), Math.max(node.size[1], 780)];
    cleanupExtraWidgets();
    node.setDirtyCanvas(true, true);

    setTimeout(syncExternalConnectionUI, 150);
    updateTimelineDisplay();
}
