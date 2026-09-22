import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Helper for building API URLs
const getApiURL = (path) => {
    if (api && typeof api.apiURL === "function") {
        return api.apiURL(path);
    }
    return path;
};

// Global editor modal singleton
let editorModalInstance = null;

class VideoMaskEditorDialog {
    constructor() {
        this.nodeId = null;
        this.sessionData = null;
        this.currentFrame = 0;
        this.numFrames = 0;
        this.width = 0;
        this.height = 0;
        this.isPaused = false;
        
        // Tool state
        this.currentTool = "brush"; // "brush", "eraser", "move", "pan"
        this.brushSize = 30;
        this.eraserSize = 30;
        this.maskOpacity = 0.55;
        this.maskColor = "#00f0ff"; // default cyan
        this.viewMode = "overlay"; // "overlay", "mask", "video"
        
        // Pan & Zoom
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;
        
        // Drawing state
        this.isDrawing = false;
        this.lastDrawX = 0;
        this.lastDrawY = 0;
        this.undoStack = [];
        this.redoStack = [];
        
        // Move mask state
        this.isMovingMask = false;
        this.moveStartX = 0;
        this.moveStartY = 0;
        this.moveAccumX = 0;
        this.moveAccumY = 0;
        
        // Playback
        this.isPlaying = false;
        this.playFps = 12;
        this.playTimer = null;
        
        // Frame caches: Map<frameIdx, Image / HTMLCanvasElement>
        this.imageCache = new Map();
        this.maskCanvasCache = new Map();
        this.editedIndices = new Set();
        this.flickerIndices = [];
        
        this.createDOM();
        this.setupShortcuts();
    }
    
    createDOM() {
        // Overlay backdrop container
        this.backdrop = document.createElement("div");
        this.backdrop.id = "vme-backdrop";
        this.backdrop.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(8, 10, 14, 0.9); backdrop-filter: blur(8px);
            z-index: 999999; display: none; align-items: center; justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #e6edf3; user-select: none;
        `;
        
        // Main window
        this.window = document.createElement("div");
        this.window.style.cssText = `
            width: 95vw; height: 92vh; background: #16191f; border: 1px solid #30363d;
            border-radius: 10px; box-shadow: 0 20px 60px rgba(0,0,0,0.9);
            display: flex; flex-direction: column; overflow: hidden;
        `;
        this.backdrop.appendChild(this.window);
        
        // 1. Header Bar
        this.header = document.createElement("div");
        this.header.style.cssText = `
            height: 48px; background: #1b1f27; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 16px; flex-shrink: 0;
        `;
        
        this.headerLeft = document.createElement("div");
        this.headerLeft.style.cssText = "display: flex; align-items: center; gap: 14px;";
        
        const titleSpan = document.createElement("span");
        titleSpan.style.cssText = "font-size: 16px; font-weight: 600; color: #58a6ff; display: flex; align-items: center; gap: 6px;";
        titleSpan.innerHTML = `🎬 视频遮罩修复编辑器 <span style="font-size: 12px; color: #8b949e; font-weight: normal;">(Video Mask Editor)</span>`;
        this.headerLeft.appendChild(titleSpan);
        
        this.headerInfo = document.createElement("span");
        this.headerInfo.style.cssText = "font-size: 13px; color: #8b949e;";
        this.headerInfo.textContent = "准备就绪";
        this.headerLeft.appendChild(this.headerInfo);
        
        this.statusBadge = document.createElement("span");
        this.statusBadge.style.cssText = "display: none; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;";
        this.headerLeft.appendChild(this.statusBadge);
        
        this.headerRight = document.createElement("div");
        this.headerRight.style.cssText = "display: flex; align-items: center; gap: 10px;";
        
        // "Continue" button (green, prominent when paused)
        this.btnContinue = document.createElement("button");
        this.btnContinue.id = "vme-btn-continue";
        this.btnContinue.style.cssText = `
            background: #238636; color: #fff; border: 1px solid rgba(240,246,252,0.1);
            padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600;
            cursor: pointer; display: flex; align-items: center; gap: 6px;
            transition: background 0.15s;
        `;
        this.btnContinue.innerHTML = `▶️ 保存并继续运行`;
        this.btnContinue.onclick = () => this.onSaveAndContinue();
        
        // "Save Edits" button
        this.btnSave = document.createElement("button");
        this.btnSave.style.cssText = `
            background: #1f6feb; color: #fff; border: 1px solid rgba(240,246,252,0.1);
            padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 500;
            cursor: pointer; display: flex; align-items: center; gap: 6px;
        `;
        this.btnSave.innerHTML = `💾 保存编辑`;
        this.btnSave.onclick = () => this.saveCurrentFrameMask(true);
        
        // Close button
        this.btnClose = document.createElement("button");
        this.btnClose.style.cssText = `
            background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
            padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
        `;
        this.btnClose.innerHTML = `✖ 关闭`;
        this.btnClose.onclick = () => this.close();
        
        this.headerRight.appendChild(this.btnContinue);
        this.headerRight.appendChild(this.btnSave);
        this.headerRight.appendChild(this.btnClose);
        this.header.appendChild(this.headerLeft);
        this.header.appendChild(this.headerRight);
        this.window.appendChild(this.header);
        
        // 2. Middle Body: Left Toolbar + Center Canvas
        this.body = document.createElement("div");
        this.body.style.cssText = "display: flex; flex: 1; overflow: hidden; position: relative;";
        this.window.appendChild(this.body);
        
        // Center Canvas Viewport (CREATED BEFORE TOOLBAR TO PREVENT NULL REFS)
        this.viewport = document.createElement("div");
        this.viewport.style.cssText = `
            flex: 1; background: #0b0d11; position: relative; overflow: hidden; cursor: crosshair;
        `;
        
        // Canvas Container (for pan & zoom)
        this.canvasContainer = document.createElement("div");
        this.canvasContainer.style.cssText = `
            position: absolute; top: 0; left: 0; transform-origin: 0 0;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
        `;
        this.viewport.appendChild(this.canvasContainer);
        
        // Bottom canvas: video frame image
        this.videoCanvas = document.createElement("canvas");
        this.videoCanvas.style.cssText = "position: absolute; top: 0; left: 0;";
        this.canvasContainer.appendChild(this.videoCanvas);
        this.videoCtx = this.videoCanvas.getContext("2d");
        
        // Top canvas: mask (editable)
        this.maskCanvas = document.createElement("canvas");
        this.maskCanvas.style.cssText = "position: absolute; top: 0; left: 0;";
        this.canvasContainer.appendChild(this.maskCanvas);
        this.maskCtx = this.maskCanvas.getContext("2d", { willReadFrequently: true });
        
        // Cursor preview overlay
        this.cursorCircle = document.createElement("div");
        this.cursorCircle.style.cssText = `
            position: absolute; border: 1.5px solid #fff; border-radius: 50%;
            pointer-events: none; transform: translate(-50%, -50%); display: none;
            box-shadow: 0 0 2px #000; z-index: 100;
        `;
        this.viewport.appendChild(this.cursorCircle);
        
        // Floating move offset badge
        this.moveBadge = document.createElement("div");
        this.moveBadge.style.cssText = `
            position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
            background: rgba(31, 111, 235, 0.9); color: #fff; padding: 4px 12px;
            border-radius: 20px; font-size: 13px; font-weight: 500; display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5); pointer-events: none; z-index: 100;
        `;
        this.viewport.appendChild(this.moveBadge);
        
        // Left Toolbar
        this.toolbar = document.createElement("div");
        this.toolbar.style.cssText = `
            width: 200px; background: #1b1f27; border-right: 1px solid #30363d;
            padding: 12px; display: flex; flex-direction: column; gap: 12px;
            overflow-y: auto; flex-shrink: 0; font-size: 12px;
        `;
        this.body.appendChild(this.toolbar);
        this.body.appendChild(this.viewport);
        
        this.buildToolbar();
        this.setupCanvasEvents();
        
        // 3. Bottom Timeline & Transport Bar
        this.bottomBar = document.createElement("div");
        this.bottomBar.style.cssText = `
            height: 105px; background: #1b1f27; border-top: 1px solid #30363d;
            padding: 8px 16px; display: flex; flex-direction: column; gap: 6px; flex-shrink: 0;
        `;
        this.window.appendChild(this.bottomBar);
        
        this.buildTimeline();
        document.body.appendChild(this.backdrop);
    }
    
    buildToolbar() {
        const createToolBtn = (id, icon, label, shortcut) => {
            const btn = document.createElement("button");
            btn.dataset.tool = id;
            btn.style.cssText = `
                width: 100%; height: 32px; background: #21262d; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 6px; font-size: 12px;
                display: flex; align-items: center; justify-content: space-between;
                padding: 0 10px; cursor: pointer; transition: all 0.15s;
            `;
            btn.innerHTML = `<span>${icon} ${label}</span><kbd style="background:#16191f;padding:1px 4px;border-radius:3px;font-size:10px;color:#8b949e;">${shortcut}</kbd>`;
            btn.onclick = () => this.setTool(id);
            return btn;
        };
        
        // Tool group
        const toolGroup = document.createElement("div");
        toolGroup.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
        toolGroup.innerHTML = `<div style="font-weight: 600; color: #8b949e; margin-bottom: 2px;">修补工具</div>`;
        
        this.btnBrush = createToolBtn("brush", "🖌️", "画笔 (涂抹)", "B");
        this.btnEraser = createToolBtn("eraser", "🧹", "橡皮擦 (擦除)", "E");
        this.btnMove = createToolBtn("move", "✥", "拖动遮罩", "M");
        this.btnPan = createToolBtn("pan", "✋", "画布平移", "H");
        
        toolGroup.appendChild(this.btnBrush);
        toolGroup.appendChild(this.btnEraser);
        toolGroup.appendChild(this.btnMove);
        toolGroup.appendChild(this.btnPan);
        this.toolbar.appendChild(toolGroup);
        
        // Frame Clone & Repair actions
        const actionGroup = document.createElement("div");
        actionGroup.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
        actionGroup.innerHTML = `<div style="font-weight: 600; color: #8b949e; margin-bottom: 2px;">帧克隆与操作</div>`;
        
        const createActionBtn = (icon, label, shortcut, onClick, color="#21262d") => {
            const btn = document.createElement("button");
            btn.style.cssText = `
                width: 100%; height: 30px; background: ${color}; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 6px; font-size: 12px;
                display: flex; align-items: center; justify-content: space-between;
                padding: 0 10px; cursor: pointer;
            `;
            btn.innerHTML = `<span>${icon} ${label}</span><kbd style="background:#16191f;padding:1px 4px;border-radius:3px;font-size:10px;color:#8b949e;">${shortcut}</kbd>`;
            btn.onclick = onClick;
            return btn;
        };
        
        const btnCopyPrev = createActionBtn("📋", "继承上一帧", "P", () => this.copyPrevFrame());
        const btnCopyNext = createActionBtn("📋", "继承下一帧", "N", () => this.copyNextFrame());
        const btnSmooth = createActionBtn("🔄", "平滑/羽化", "S", () => this.smoothCurrentMask());
        const btnClear = createActionBtn("🗑️", "清空当前帧", "Del", () => this.clearCurrentMask());
        const btnReset = createActionBtn("🔁", "重置当前帧", "R", () => this.resetCurrentFrame(), "#2d2020");
        const btnResetAll = createActionBtn("⚠️", "重置所有帧", "全部", () => this.resetAllFrames(), "#3d1818");
        btnResetAll.style.borderColor = "#da3633";
        btnResetAll.style.color = "#f85149";
        
        actionGroup.appendChild(btnCopyPrev);
        actionGroup.appendChild(btnCopyNext);
        actionGroup.appendChild(btnSmooth);
        actionGroup.appendChild(btnClear);
        actionGroup.appendChild(btnReset);
        actionGroup.appendChild(btnResetAll);
        this.toolbar.appendChild(actionGroup);
        
        // Tool settings
        const settingsGroup = document.createElement("div");
        settingsGroup.style.cssText = "display: flex; flex-direction: column; gap: 8px;";
        settingsGroup.innerHTML = `<div style="font-weight: 600; color: #8b949e;">参数设置</div>`;
        
        // Brush size slider
        const sizeRow = document.createElement("div");
        sizeRow.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>画笔大小:</span><span id="vme-size-val">${this.brushSize}px</span>
            </div>
            <input type="range" id="vme-size-slider" min="2" max="150" value="${this.brushSize}" style="width:100%;">
        `;
        settingsGroup.appendChild(sizeRow);
        
        // Mask opacity slider
        const opacityRow = document.createElement("div");
        opacityRow.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>遮罩透明度:</span><span id="vme-opacity-val">${Math.round(this.maskOpacity * 100)}%</span>
            </div>
            <input type="range" id="vme-opacity-slider" min="10" max="100" value="${Math.round(this.maskOpacity * 100)}" style="width:100%;">
        `;
        settingsGroup.appendChild(opacityRow);
        
        // Mask color choices
        const colorRow = document.createElement("div");
        colorRow.innerHTML = `<div style="margin-bottom: 4px;">遮罩颜色:</div>`;
        const colorContainer = document.createElement("div");
        colorContainer.style.cssText = "display: flex; gap: 6px;";
        const colors = ["#00f0ff", "#ff3366", "#00ff66", "#ffd700", "#d000ff"];
        colors.forEach(c => {
            const dot = document.createElement("div");
            dot.style.cssText = `
                width: 22px; height: 22px; border-radius: 50%; background: ${c};
                cursor: pointer; border: 2px solid ${c === this.maskColor ? '#fff' : 'transparent'};
                box-sizing: border-box; transition: transform 0.1s;
            `;
            dot.onclick = () => {
                this.maskColor = c;
                colorContainer.querySelectorAll("div").forEach(d => d.style.borderColor = "transparent");
                dot.style.borderColor = "#fff";
                this.render();
            };
            colorContainer.appendChild(dot);
        });
        colorRow.appendChild(colorContainer);
        settingsGroup.appendChild(colorRow);
        
        // View mode selector
        const viewRow = document.createElement("div");
        viewRow.innerHTML = `<div style="margin-bottom: 4px;">视窗模式:</div>`;
        const viewSelect = document.createElement("select");
        viewSelect.style.cssText = "width: 100%; height: 26px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px;";
        viewSelect.innerHTML = `
            <option value="overlay">重叠模式 (Overlay)</option>
            <option value="mask">纯遮罩 (Mask Only)</option>
            <option value="video">纯原视频 (Video Only)</option>
        `;
        viewSelect.onchange = (e) => {
            this.viewMode = e.target.value;
            this.render();
        };
        viewRow.appendChild(viewSelect);
        settingsGroup.appendChild(viewRow);
        
        // Zoom controls
        const zoomRow = document.createElement("div");
        zoomRow.style.cssText = "display: flex; gap: 6px; margin-top: 4px;";
        const btnFit = document.createElement("button");
        btnFit.style.cssText = "flex: 1; height: 26px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; cursor: pointer;";
        btnFit.textContent = "适应画布";
        btnFit.onclick = () => this.fitToScreen();
        const btn100 = document.createElement("button");
        btn100.style.cssText = "flex: 1; height: 26px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; cursor: pointer;";
        btn100.textContent = "100%";
        btn100.onclick = () => {
            this.zoom = 1.0;
            this.panX = (this.viewport.clientWidth - this.width) / 2;
            this.panY = (this.viewport.clientHeight - this.height) / 2;
            this.updateTransform();
        };
        zoomRow.appendChild(btnFit);
        zoomRow.appendChild(btn100);
        settingsGroup.appendChild(zoomRow);
        
        this.toolbar.appendChild(settingsGroup);
        
        // Slider listeners
        sizeRow.querySelector("#vme-size-slider").oninput = (e) => {
            const v = parseInt(e.target.value);
            this.brushSize = v;
            this.eraserSize = v;
            sizeRow.querySelector("#vme-size-val").textContent = `${v}px`;
            this.updateCursorCircle();
        };
        
        opacityRow.querySelector("#vme-opacity-slider").oninput = (e) => {
            const v = parseInt(e.target.value);
            this.maskOpacity = v / 100.0;
            opacityRow.querySelector("#vme-opacity-val").textContent = `${v}%`;
            this.render();
        };
        
        this.setTool("brush");
    }
    
    setTool(tool) {
        this.currentTool = tool;
        const btns = [this.btnBrush, this.btnEraser, this.btnMove, this.btnPan];
        btns.forEach(b => {
            if (!b) return;
            if (b.dataset && b.dataset.tool === tool) {
                b.style.background = "#1f6feb";
                b.style.color = "#fff";
                b.style.borderColor = "#58a6ff";
            } else {
                b.style.background = "#21262d";
                b.style.color = "#c9d1d9";
                b.style.borderColor = "#30363d";
            }
        });
        
        if (this.viewport) {
            if (tool === "pan") {
                this.viewport.style.cursor = "grab";
            } else if (tool === "move") {
                this.viewport.style.cursor = "move";
            } else {
                this.viewport.style.cursor = "crosshair";
            }
        }
        this.updateCursorCircle();
    }
    
    buildTimeline() {
        const ctrlRow = document.createElement("div");
        ctrlRow.style.cssText = "display: flex; align-items: center; justify-content: space-between;";
        
        const playGroup = document.createElement("div");
        playGroup.style.cssText = "display: flex; align-items: center; gap: 8px;";
        
        this.btnPlay = document.createElement("button");
        this.btnPlay.style.cssText = `
            width: 32px; height: 32px; background: #21262d; color: #fff;
            border: 1px solid #30363d; border-radius: 6px; font-size: 14px;
            display: flex; align-items: center; justify-content: center; cursor: pointer;
        `;
        this.btnPlay.innerHTML = "▶";
        this.btnPlay.onclick = () => this.togglePlay();
        
        const btnStepPrev = document.createElement("button");
        btnStepPrev.style.cssText = "width: 28px; height: 28px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; cursor: pointer;";
        btnStepPrev.innerHTML = "◀";
        btnStepPrev.title = "上一帧 (左箭头)";
        btnStepPrev.onclick = () => this.stepFrame(-1);
        
        const btnStepNext = document.createElement("button");
        btnStepNext.style.cssText = "width: 28px; height: 28px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; cursor: pointer;";
        btnStepNext.innerHTML = "▶";
        btnStepNext.title = "下一帧 (右箭头)";
        btnStepNext.onclick = () => this.stepFrame(1);
        
        // FPS selector
        const fpsSelect = document.createElement("select");
        fpsSelect.style.cssText = "height: 28px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; font-size: 12px;";
        [8, 12, 16, 24, 30].forEach(fps => {
            const opt = document.createElement("option");
            opt.value = fps;
            opt.textContent = `${fps} FPS`;
            if (fps === this.playFps) opt.selected = true;
            fpsSelect.appendChild(opt);
        });
        fpsSelect.onchange = (e) => {
            this.playFps = parseInt(e.target.value);
            if (this.isPlaying) {
                this.stopPlay();
                this.startPlay();
            }
        };
        
        playGroup.appendChild(this.btnPlay);
        playGroup.appendChild(btnStepPrev);
        playGroup.appendChild(btnStepNext);
        playGroup.appendChild(fpsSelect);
        
        // Center: Frame indicator
        this.frameIndicator = document.createElement("div");
        this.frameIndicator.style.cssText = "font-size: 13px; font-weight: 600; color: #58a6ff;";
        this.frameIndicator.textContent = "帧 1 / 1";
        
        // Right: Flicker jump alerts
        const flickerGroup = document.createElement("div");
        flickerGroup.style.cssText = "display: flex; align-items: center; gap: 8px;";
        
        this.flickerBadge = document.createElement("span");
        this.flickerBadge.style.cssText = `
            display: none; background: rgba(218, 54, 51, 0.2); color: #f85149;
            border: 1px solid #da3633; padding: 2px 8px; border-radius: 4px; font-size: 12px;
        `;
        
        this.btnNextFlicker = document.createElement("button");
        this.btnNextFlicker.style.cssText = `
            display: none; background: #21262d; color: #f85149; border: 1px solid #da3633;
            padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer;
        `;
        this.btnNextFlicker.innerHTML = "⚠️ 下一处闪烁帧";
        this.btnNextFlicker.onclick = () => this.jumpNextFlicker();
        
        flickerGroup.appendChild(this.flickerBadge);
        flickerGroup.appendChild(this.btnNextFlicker);
        
        ctrlRow.appendChild(playGroup);
        ctrlRow.appendChild(this.frameIndicator);
        ctrlRow.appendChild(flickerGroup);
        this.bottomBar.appendChild(ctrlRow);
        
        // Slider scrubber
        this.scrubber = document.createElement("input");
        this.scrubber.type = "range";
        this.scrubber.min = "0";
        this.scrubber.max = "0";
        this.scrubber.value = "0";
        this.scrubber.style.cssText = "width: 100%; height: 6px; cursor: pointer; margin: 4px 0;";
        this.scrubber.oninput = (e) => {
            this.goToFrame(parseInt(e.target.value));
        };
        this.bottomBar.appendChild(this.scrubber);
        
        // Visual Track Strip (showing edited & flicker marks)
        this.trackContainer = document.createElement("div");
        this.trackContainer.style.cssText = `
            width: 100%; height: 26px; background: #16191f; border: 1px solid #30363d;
            border-radius: 4px; position: relative; overflow: hidden; display: flex;
        `;
        this.bottomBar.appendChild(this.trackContainer);
    }
    
    updateTrackStrip() {
        if (!this.trackContainer) return;
        this.trackContainer.innerHTML = "";
        if (this.numFrames <= 0) return;
        
        for (let i = 0; i < this.numFrames; i++) {
            const cell = document.createElement("div");
            cell.dataset.frame = i;
            cell.style.cssText = `
                flex: 1; min-width: 1px; height: 100%; position: relative;
                cursor: pointer; border-right: 1px solid rgba(255,255,255,0.04);
            `;
            
            // Highlight current frame
            if (i === this.currentFrame) {
                cell.style.background = "rgba(88, 166, 255, 0.4)";
                cell.style.border = "1px solid #58a6ff";
            }
            
            // Mark edited frames with green dot
            if (this.editedIndices.has(i)) {
                const dot = document.createElement("div");
                dot.style.cssText = `
                    position: absolute; bottom: 2px; left: 50%; transform: translateX(-50%);
                    width: 5px; height: 5px; border-radius: 50%; background: #3fb950;
                `;
                cell.appendChild(dot);
            }
            
            // Mark flicker frame with red warning indicator
            if (this.flickerIndices.includes(i)) {
                cell.style.background = "rgba(218, 54, 51, 0.35)";
                const warn = document.createElement("div");
                warn.style.cssText = `
                    position: absolute; top: 1px; left: 50%; transform: translateX(-50%);
                    font-size: 10px; line-height: 1; color: #f85149;
                `;
                warn.textContent = "!";
                cell.appendChild(warn);
            }
            
            cell.onclick = () => this.goToFrame(i);
            this.trackContainer.appendChild(cell);
        }
    }
    
    setupCanvasEvents() {
        if (!this.viewport) return;
        
        // Wheel to Zoom
        this.viewport.addEventListener("wheel", (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
            const newZoom = Math.max(0.1, Math.min(10.0, this.zoom * zoomFactor));
            
            const rect = this.viewport.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
            this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            
            this.updateTransform();
        }, { passive: false });
        
        // Mouse Down
        this.viewport.addEventListener("mousedown", (e) => {
            if (e.button === 1 || this.currentTool === "pan" || e.altKey) {
                this.isPanning = true;
                this.panStartX = e.clientX - this.panX;
                this.panStartY = e.clientY - this.panY;
                this.viewport.style.cursor = "grabbing";
                return;
            }
            
            if (e.button !== 0) return;
            
            const pt = this.getCanvasCoords(e);
            
            if (this.currentTool === "move") {
                this.isMovingMask = true;
                this.moveStartX = pt.x;
                this.moveStartY = pt.y;
                this.moveAccumX = 0;
                this.moveAccumY = 0;
                if (this.moveBadge) {
                    this.moveBadge.style.display = "block";
                    this.moveBadge.textContent = "位移: X: 0, Y: 0";
                }
                this.saveUndoState();
            } else if (this.currentTool === "brush" || this.currentTool === "eraser") {
                this.isDrawing = true;
                this.lastDrawX = pt.x;
                this.lastDrawY = pt.y;
                this.saveUndoState();
                this.drawPoint(pt.x, pt.y);
            }
        });
        
        // Mouse Move
        window.addEventListener("mousemove", (e) => {
            this.updateCursorPosition(e);
            
            if (this.isPanning) {
                this.panX = e.clientX - this.panStartX;
                this.panY = e.clientY - this.panStartY;
                this.updateTransform();
                return;
            }
            
            if (this.isMovingMask) {
                const pt = this.getCanvasCoords(e);
                const dx = Math.round(pt.x - this.moveStartX);
                const dy = Math.round(pt.y - this.moveStartY);
                this.moveAccumX = dx;
                this.moveAccumY = dy;
                if (this.moveBadge) {
                    this.moveBadge.textContent = `位移: X: ${dx > 0 ? '+' : ''}${dx}px, Y: ${dy > 0 ? '+' : ''}${dy}px`;
                }
                return;
            }
            
            if (this.isDrawing) {
                const pt = this.getCanvasCoords(e);
                this.drawLine(this.lastDrawX, this.lastDrawY, pt.x, pt.y);
                this.lastDrawX = pt.x;
                this.lastDrawY = pt.y;
            }
        });
        
        // Mouse Up
        window.addEventListener("mouseup", (e) => {
            if (this.isPanning) {
                this.isPanning = false;
                if (this.viewport) this.viewport.style.cursor = this.currentTool === "pan" ? "grab" : "crosshair";
            }
            
            if (this.isMovingMask) {
                this.isMovingMask = false;
                if (this.moveBadge) this.moveBadge.style.display = "none";
                if (this.moveAccumX !== 0 || this.moveAccumY !== 0) {
                    this.applyMaskTranslation(this.moveAccumX, this.moveAccumY);
                }
            }
            
            if (this.isDrawing) {
                this.isDrawing = false;
                this.editedIndices.add(this.currentFrame);
                this.saveCurrentFrameMask(false);
                this.updateTrackStrip();
            }
        });
    }
    
    getCanvasCoords(e) {
        if (!this.canvasContainer) return { x: 0, y: 0 };
        const rect = this.canvasContainer.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / this.zoom,
            y: (e.clientY - rect.top) / this.zoom
        };
    }
    
    updateTransform() {
        if (!this.canvasContainer) return;
        this.canvasContainer.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
    }
    
    fitToScreen() {
        if (!this.width || !this.height || !this.viewport) return;
        const vWidth = Math.max(100, this.viewport.clientWidth - 60);
        const vHeight = Math.max(100, this.viewport.clientHeight - 60);
        const scale = Math.min(vWidth / this.width, vHeight / this.height);
        this.zoom = Math.max(0.05, Math.min(10.0, scale));
        this.panX = (this.viewport.clientWidth - this.width * this.zoom) / 2;
        this.panY = (this.viewport.clientHeight - this.height * this.zoom) / 2;
        this.updateTransform();
    }
    
    updateCursorPosition(e) {
        if (!this.cursorCircle || !this.viewport) return;
        const rect = this.viewport.getBoundingClientRect();
        if (e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom) {
            if (this.currentTool === "brush" || this.currentTool === "eraser") {
                this.cursorCircle.style.display = "block";
                this.cursorCircle.style.left = `${e.clientX - rect.left}px`;
                this.cursorCircle.style.top = `${e.clientY - rect.top}px`;
                const radius = (this.currentTool === "brush" ? this.brushSize : this.eraserSize) * this.zoom;
                this.cursorCircle.style.width = `${radius * 2}px`;
                this.cursorCircle.style.height = `${radius * 2}px`;
                return;
            }
        }
        this.cursorCircle.style.display = "none";
    }
    
    updateCursorCircle() {
        if (!this.cursorCircle) return;
        const radius = (this.currentTool === "brush" ? this.brushSize : this.eraserSize) * this.zoom;
        this.cursorCircle.style.width = `${radius * 2}px`;
        this.cursorCircle.style.height = `${radius * 2}px`;
    }
    
    drawPoint(x, y) {
        if (!this.maskCtx) return;
        const isEraser = this.currentTool === "eraser";
        const radius = isEraser ? this.eraserSize : this.brushSize;
        
        this.maskCtx.save();
        this.maskCtx.globalCompositeOperation = isEraser ? "destination-out" : "source-over";
        this.maskCtx.fillStyle = "#ffffff";
        this.maskCtx.beginPath();
        this.maskCtx.arc(x, y, radius, 0, Math.PI * 2);
        this.maskCtx.fill();
        this.maskCtx.restore();
        
        this.render();
    }
    
    drawLine(x1, y1, x2, y2) {
        if (!this.maskCtx) return;
        const isEraser = this.currentTool === "eraser";
        const radius = isEraser ? this.eraserSize : this.brushSize;
        
        this.maskCtx.save();
        this.maskCtx.globalCompositeOperation = isEraser ? "destination-out" : "source-over";
        this.maskCtx.strokeStyle = "#ffffff";
        this.maskCtx.lineWidth = radius * 2;
        this.maskCtx.lineCap = "round";
        this.maskCtx.lineJoin = "round";
        this.maskCtx.beginPath();
        this.maskCtx.moveTo(x1, y1);
        this.maskCtx.lineTo(x2, y2);
        this.maskCtx.stroke();
        this.maskCtx.restore();
        
        this.render();
    }
    
    saveUndoState() {
        if (!this.width || !this.height || !this.maskCtx) return;
        const imgData = this.maskCtx.getImageData(0, 0, this.width, this.height);
        this.undoStack.push(imgData);
        if (this.undoStack.length > 20) this.undoStack.shift();
        this.redoStack = [];
    }
    
    undo() {
        if (this.undoStack.length === 0 || !this.maskCtx) return;
        const currentData = this.maskCtx.getImageData(0, 0, this.width, this.height);
        this.redoStack.push(currentData);
        const prevState = this.undoStack.pop();
        this.maskCtx.putImageData(prevState, 0, 0);
        this.render();
        this.saveCurrentFrameMask(false);
    }
    
    redo() {
        if (this.redoStack.length === 0 || !this.maskCtx) return;
        const currentData = this.maskCtx.getImageData(0, 0, this.width, this.height);
        this.undoStack.push(currentData);
        const nextState = this.redoStack.pop();
        this.maskCtx.putImageData(nextState, 0, 0);
        this.render();
        this.saveCurrentFrameMask(false);
    }
    
    applyMaskTranslation(dx, dy) {
        if (!this.maskCanvas || !this.maskCtx) return;
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = this.width;
        tempCanvas.height = this.height;
        const tempCtx = tempCanvas.getContext("2d");
        tempCtx.drawImage(this.maskCanvas, 0, 0);
        
        this.maskCtx.clearRect(0, 0, this.width, this.height);
        this.maskCtx.drawImage(tempCanvas, dx, dy);
        this.render();
        
        this.editedIndices.add(this.currentFrame);
        this.updateTrackStrip();
        
        fetch(getApiURL("/video_mask_editor/translate_mask"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: this.nodeId,
                frame_idx: this.currentFrame,
                dx: dx,
                dy: dy
            })
        }).catch(err => console.error("Translate mask error:", err));
    }
    
    copyPrevFrame() {
        if (this.currentFrame <= 0) return;
        this.copyFromFrame(this.currentFrame - 1);
    }
    
    copyNextFrame() {
        if (this.currentFrame >= this.numFrames - 1) return;
        this.copyFromFrame(this.currentFrame + 1);
    }
    
    async copyFromFrame(sourceIdx) {
        this.saveUndoState();
        const srcCanvas = await this.getOrLoadMaskCanvas(sourceIdx);
        if (srcCanvas && this.maskCtx) {
            this.maskCtx.clearRect(0, 0, this.width, this.height);
            this.maskCtx.drawImage(srcCanvas, 0, 0);
            this.render();
            
            this.editedIndices.add(this.currentFrame);
            this.updateTrackStrip();
            
            fetch(getApiURL("/video_mask_editor/copy_frame"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: this.nodeId,
                    target_frame: this.currentFrame,
                    source_frame: sourceIdx
                })
            }).catch(err => console.error("Copy frame error:", err));
        }
    }
    
    smoothCurrentMask() {
        if (!this.maskCtx) return;
        this.saveUndoState();
        const imgData = this.maskCtx.getImageData(0, 0, this.width, this.height);
        const data = imgData.data;
        const copy = new Uint8Array(data);
        const w = this.width;
        const h = this.height;
        
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = (y * w + x) * 4;
                let sum = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        sum += copy[((y + dy) * w + (x + dx)) * 4 + 3];
                    }
                }
                data[idx + 3] = sum / 9.0;
            }
        }
        this.maskCtx.putImageData(imgData, 0, 0);
        this.render();
        this.saveCurrentFrameMask(false);
    }
    
    clearCurrentMask() {
        if (!this.maskCtx) return;
        this.saveUndoState();
        this.maskCtx.clearRect(0, 0, this.width, this.height);
        this.render();
        this.editedIndices.add(this.currentFrame);
        this.updateTrackStrip();
        this.saveCurrentFrameMask(false);
    }
    
    async resetCurrentFrame() {
        this.saveUndoState();
        try {
            const resp = await fetch(getApiURL("/video_mask_editor/reset_frame"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: this.nodeId,
                    frame_idx: this.currentFrame
                })
            });
            const data = await resp.json();
            if (data.success) {
                this.editedIndices.delete(this.currentFrame);
                this.maskCanvasCache.delete(this.currentFrame);
                await this.loadCurrentFrame();
                this.updateTrackStrip();
            }
        } catch (e) {
            console.error("Reset frame error:", e);
        }
    }
    
    async resetAllFrames() {
        if (!confirm("⚠️ 确定要清空该视频的所有手动编辑，恢复到初始输入遮罩吗？\n此操作将清除所有已保存的手动画笔和位移。")) {
            return;
        }
        try {
            const resp = await fetch(getApiURL("/video_mask_editor/reset_all"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: this.nodeId
                })
            });
            const data = await resp.json();
            if (data.success) {
                this.editedIndices.clear();
                this.maskCanvasCache.clear();
                this.imageCache.clear();
                this.undoStack = [];
                this.redoStack = [];
                await this.loadCurrentFrame();
                this.updateTrackStrip();
                alert("已成功恢复初始遮罩！");
            } else {
                alert("重置失败: " + (data.error || "未知错误"));
            }
        } catch (e) {
            console.error("Reset all frames error:", e);
            alert("请求重置失败: " + e.message);
        }
    }
    
    setupShortcuts() {
        window.addEventListener("keydown", (e) => {
            if (!this.backdrop || this.backdrop.style.display !== "flex") return;
            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
            
            if (e.key === "b" || e.key === "B") this.setTool("brush");
            else if (e.key === "e" || e.key === "E") this.setTool("eraser");
            else if (e.key === "m" || e.key === "M") this.setTool("move");
            else if (e.key === "h" || e.key === "H") this.setTool("pan");
            else if (e.key === "[" || e.key === "p" || e.key === "P") this.copyPrevFrame();
            else if (e.key === "]" || e.key === "n" || e.key === "N") this.copyNextFrame();
            else if (e.key === "ArrowLeft") this.stepFrame(-1);
            else if (e.key === "ArrowRight") this.stepFrame(1);
            else if (e.key === " " || e.code === "Space") {
                e.preventDefault();
                this.togglePlay();
            } else if (e.ctrlKey && (e.key === "z" || e.key === "Z")) {
                e.preventDefault();
                if (e.shiftKey) this.redo();
                else this.undo();
            } else if (e.ctrlKey && (e.key === "y" || e.key === "Y")) {
                e.preventDefault();
                this.redo();
            } else if (e.key === "Escape") {
                this.close();
            }
        });
    }
    
    async open(nodeId) {
        this.nodeId = nodeId;
        if (!this.backdrop) this.createDOM();
        
        this.backdrop.style.display = "flex";
        this.backdrop.style.zIndex = "999999";
        
        // Reset state
        this.imageCache.clear();
        this.maskCanvasCache.clear();
        this.undoStack = [];
        this.redoStack = [];
        this.currentFrame = 0;
        
        try {
            console.log(`[VideoMaskEditor] Fetching session for node: ${nodeId}...`);
            const resp = await fetch(getApiURL(`/video_mask_editor/session?node_id=${nodeId}`));
            const data = await resp.json();
            
            if (!data.has_session) {
                alert(`未找到该节点的运行会话数据 (节点ID: ${nodeId})。\n请先运行包含该节点的工作流。`);
                this.close();
                return;
            }
            
            this.sessionData = data;
            this.nodeId = data.node_id; // Sync resolved node ID
            this.numFrames = data.num_frames;
            this.width = data.width;
            this.height = data.height;
            this.isPaused = data.status === "waiting_for_user";
            this.editedIndices = new Set(data.edited_indices || []);
            this.flickerIndices = data.flicker_indices || [];
            
            // Resize canvases
            this.videoCanvas.width = this.width;
            this.videoCanvas.height = this.height;
            this.maskCanvas.width = this.width;
            this.maskCanvas.height = this.height;
            this.canvasContainer.style.width = `${this.width}px`;
            this.canvasContainer.style.height = `${this.height}px`;
            
            // Update UI headers
            if (this.headerInfo) {
                this.headerInfo.textContent = `分辨率: ${this.width}×${this.height} | 总帧数: ${this.numFrames} 帧`;
            }
            
            if (this.statusBadge) {
                if (this.isPaused) {
                    this.statusBadge.style.display = "inline-block";
                    this.statusBadge.style.background = "#238636";
                    this.statusBadge.style.color = "#fff";
                    this.statusBadge.textContent = "⏸️ 执行已暂停 - 请修复遮罩后点击继续";
                    if (this.btnContinue) this.btnContinue.style.display = "flex";
                } else {
                    this.statusBadge.style.display = "inline-block";
                    this.statusBadge.style.background = "#30363d";
                    this.statusBadge.style.color = "#8b949e";
                    this.statusBadge.textContent = "⏹️ 缓存就绪";
                    if (this.btnContinue) this.btnContinue.style.display = "none";
                }
            }
            
            // Flicker alerts
            if (this.flickerIndices.length > 0) {
                if (this.flickerBadge) {
                    this.flickerBadge.style.display = "inline-block";
                    this.flickerBadge.textContent = `⚠️ 检测到 ${this.flickerIndices.length} 处疑似闪烁缺失`;
                }
                if (this.btnNextFlicker) this.btnNextFlicker.style.display = "inline-block";
            } else {
                if (this.flickerBadge) this.flickerBadge.style.display = "none";
                if (this.btnNextFlicker) this.btnNextFlicker.style.display = "none";
            }
            
            // Update scrubber
            if (this.scrubber) {
                this.scrubber.max = Math.max(0, this.numFrames - 1);
                this.scrubber.value = "0";
            }
            
            // Fit to screen
            setTimeout(() => this.fitToScreen(), 50);
            
            // Update timeline strip
            this.updateTrackStrip();
            
            // Load frame 0
            await this.goToFrame(0);
            
        } catch (err) {
            console.error("Failed to load session:", err);
            alert("加载会话失败：" + err.message);
            this.close();
        }
    }
    
    close() {
        this.stopPlay();
        if (this.backdrop) this.backdrop.style.display = "none";
    }
    
    async onSaveAndContinue() {
        await this.saveCurrentFrameMask(true);
        try {
            await fetch(getApiURL("/video_mask_editor/continue"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: this.nodeId })
            });
            this.close();
        } catch (e) {
            console.error("Failed to continue:", e);
            alert("继续运行失败: " + e.message);
        }
    }
    
    async saveCurrentFrameMask(notify = false) {
        if (!this.width || !this.height || !this.nodeId || !this.maskCanvas) return;
        
        const copyCanvas = document.createElement("canvas");
        copyCanvas.width = this.width;
        copyCanvas.height = this.height;
        copyCanvas.getContext("2d").drawImage(this.maskCanvas, 0, 0);
        this.maskCanvasCache.set(this.currentFrame, copyCanvas);
        
        const b64 = this.maskCanvas.toDataURL("image/png");
        
        if (notify && this.btnSave) {
            this.btnSave.innerHTML = "⏳ 正在保存...";
        }
        
        try {
            const resp = await fetch(getApiURL("/video_mask_editor/save_frame"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: this.nodeId,
                    frame_idx: this.currentFrame,
                    mask_data: b64
                })
            });
            const res = await resp.json();
            if (res.success) {
                this.editedIndices = new Set(res.edited_indices);
                this.updateTrackStrip();
                if (notify && this.btnSave) {
                    const origText = "💾 保存编辑";
                    this.btnSave.style.background = "#238636";
                    this.btnSave.innerHTML = "✅ 保存成功！";
                    setTimeout(() => {
                        if (this.btnSave) {
                            this.btnSave.style.background = "#1f6feb";
                            this.btnSave.innerHTML = origText;
                        }
                    }, 1500);
                }
            } else {
                console.error("Save frame error:", res.error);
                if (notify) {
                    alert("保存失败: " + (res.error || "服务器错误"));
                    if (this.btnSave) this.btnSave.innerHTML = "💾 保存编辑";
                }
            }
        } catch (err) {
            console.error("Save frame network error:", err);
            if (notify) {
                alert("网络请求失败: " + err.message);
                if (this.btnSave) this.btnSave.innerHTML = "💾 保存编辑";
            }
        }
    }
    
    async getOrLoadImage(idx) {
        if (this.imageCache.has(idx)) return this.imageCache.get(idx);
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                this.imageCache.set(idx, img);
                resolve(img);
            };
            img.onerror = () => resolve(null);
            img.src = getApiURL(`/video_mask_editor/frame?node_id=${this.nodeId}&frame_idx=${idx}&type=image&rand=${Math.random()}`);
        });
    }
    
    async getOrLoadMaskCanvas(idx) {
        if (this.maskCanvasCache.has(idx)) return this.maskCanvasCache.get(idx);
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const c = document.createElement("canvas");
                c.width = this.width;
                c.height = this.height;
                const ctx = c.getContext("2d");
                
                ctx.drawImage(img, 0, 0);
                const idata = ctx.getImageData(0, 0, this.width, this.height);
                const d = idata.data;
                for (let i = 0; i < d.length; i += 4) {
                    const val = d[i];
                    d[i] = 255;
                    d[i + 1] = 255;
                    d[i + 2] = 255;
                    d[i + 3] = val;
                }
                ctx.putImageData(idata, 0, 0);
                this.maskCanvasCache.set(idx, c);
                resolve(c);
            };
            img.onerror = () => resolve(null);
            img.src = getApiURL(`/video_mask_editor/frame?node_id=${this.nodeId}&frame_idx=${idx}&type=mask&rand=${Math.random()}`);
        });
    }
    
    async goToFrame(idx) {
        if (idx < 0 || idx >= this.numFrames) return;
        
        if (this.maskCanvas && this.maskCanvas.width > 0 && this.isDrawing) {
            await this.saveCurrentFrameMask(false);
        }
        
        this.currentFrame = idx;
        if (this.scrubber) this.scrubber.value = idx.toString();
        if (this.frameIndicator) {
            this.frameIndicator.textContent = `帧 ${idx + 1} / ${this.numFrames} (索引: ${idx})`;
        }
        this.updateTrackStrip();
        await this.loadCurrentFrame();
    }
    
    async loadCurrentFrame() {
        const [img, maskC] = await Promise.all([
            this.getOrLoadImage(this.currentFrame),
            this.getOrLoadMaskCanvas(this.currentFrame)
        ]);
        
        if (this.videoCtx) {
            this.videoCtx.clearRect(0, 0, this.width, this.height);
            if (img) this.videoCtx.drawImage(img, 0, 0);
        }
        
        if (this.maskCtx) {
            this.maskCtx.clearRect(0, 0, this.width, this.height);
            if (maskC) this.maskCtx.drawImage(maskC, 0, 0);
        }
        
        this.undoStack = [];
        this.redoStack = [];
        this.render();
    }
    
    render() {
        if (!this.videoCanvas || !this.maskCanvas) return;
        
        if (this.viewMode === "video") {
            this.videoCanvas.style.display = "block";
            this.maskCanvas.style.display = "none";
        } else if (this.viewMode === "mask") {
            this.videoCanvas.style.display = "none";
            this.maskCanvas.style.display = "block";
            this.maskCanvas.style.opacity = "1.0";
            this.maskCanvas.style.filter = "none";
        } else { // "overlay"
            this.videoCanvas.style.display = "block";
            this.maskCanvas.style.display = "block";
            this.maskCanvas.style.opacity = this.maskOpacity.toString();
            this.maskCanvas.style.filter = `drop-shadow(0 0 0 ${this.maskColor})`;
        }
    }
    
    stepFrame(delta) {
        const next = Math.max(0, Math.min(this.numFrames - 1, this.currentFrame + delta));
        this.goToFrame(next);
    }
    
    jumpNextFlicker() {
        if (this.flickerIndices.length === 0) return;
        let next = this.flickerIndices.find(f => f > this.currentFrame);
        if (next === undefined) next = this.flickerIndices[0];
        this.goToFrame(next);
    }
    
    togglePlay() {
        if (this.isPlaying) this.stopPlay();
        else this.startPlay();
    }
    
    startPlay() {
        this.isPlaying = true;
        if (this.btnPlay) this.btnPlay.innerHTML = "⏸";
        const interval = 1000 / this.playFps;
        this.playTimer = setInterval(() => {
            const next = (this.currentFrame + 1) % this.numFrames;
            this.goToFrame(next);
        }, interval);
    }
    
    stopPlay() {
        this.isPlaying = false;
        if (this.btnPlay) this.btnPlay.innerHTML = "▶";
        if (this.playTimer) {
            clearInterval(this.playTimer);
            this.playTimer = null;
        }
    }
}

// Ensure singleton instance
function getEditorDialog() {
    if (!editorModalInstance) {
        editorModalInstance = new VideoMaskEditorDialog();
    }
    return editorModalInstance;
}

// Register ComfyUI Extension
app.registerExtension({
    name: "Comfy.VideoMaskEditor",
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "VideoMaskEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                const node = this;
                
                const openEditorAction = () => {
                    try {
                        const currentId = (node.id !== undefined && node.id !== null && node.id !== -1) ? node.id.toString() : "";
                        console.log("[VideoMaskEditor] Button clicked, node ID:", currentId);
                        const editor = getEditorDialog();
                        editor.open(currentId);
                    } catch (e) {
                        console.error("[VideoMaskEditor] Error opening editor:", e);
                        alert("打开编辑器失败: " + e.message);
                    }
                };
                
                // Add Open Editor button widget
                const btnWidget = this.addWidget("button", "🎬 打开遮罩修复编辑器", "", openEditorAction);
                btnWidget.callback = openEditorAction;
                btnWidget.serialize = false;
                
                // Status label widget (read-only)
                const statusWidget = this.addWidget("text", "状态", "就绪 (等待运行)", () => {});
                statusWidget.serialize = false;
                node.statusWidget = statusWidget;
                
                this.setSize([280, 160]);
            };
        }
    },
    
    async setup() {
        api.addEventListener("video-mask-editor-update", (event) => {
            try {
                const data = event.detail || {};
                const nodeId = (data.node_id || "").toString();
                
                // Safely find the node in graph
                const node = app.graph?.getNodeById?.(nodeId) || 
                             (app.graph?._nodes || app.graph?.nodes || []).find(n => n?.id?.toString() === nodeId);
                             
                if (node && node.statusWidget) {
                    const flickerCount = (data.flicker_indices || []).length;
                    const editedCount = (data.edited_indices || []).length;
                    let text = `总帧: ${data.num_frames} | 已编辑: ${editedCount}`;
                    if (flickerCount > 0) text += ` | ⚠️ 闪烁: ${flickerCount}`;
                    if (data.is_paused) text = `⏸️ 暂停修改中 | ` + text;
                    node.statusWidget.value = text;
                    app.graph.setDirtyCanvas(true, false);
                }
                
                // If editor is currently open for this node, refresh to sync new video session
                const editor = getEditorDialog();
                if (editor.backdrop && editor.backdrop.style.display === "flex" && editor.nodeId === nodeId) {
                    editor.open(nodeId);
                } else if (data.is_paused) {
                    editor.open(nodeId);
                }
            } catch (err) {
                console.error("[VideoMaskEditor] WebSocket update handler error:", err);
            }
        });
    }
});
