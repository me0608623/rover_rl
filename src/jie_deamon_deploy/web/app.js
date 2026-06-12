/**
 * 機器人控制 Web 可視化前端
 * 使用 WebSocket 接收即時資料，Canvas 繪製可視化
 *
 * ========================================================================
 * 【差速機器人版本說明】
 * ========================================================================
 * - 搖桿只控制 vx（前後平移），vy 恆為 0（差速機器人無法橫移）
 * - 旋轉由頂部水平滑桿控制角速度 wz
 * - WebSocket 連接到 port 8890（在後端 common_types.hpp 中定義）
 *
 * 三種控制模式：
 *   mode 0 — 直接控制：搖桿 + 滑桿手動遙控
 *   mode 1 — 雷達跟隨：LiDAR 偵測目標並自動跟隨
 *   mode 2 — 導航模式：地圖導航（尚未實作）
 *
 * 速度發送限頻 10Hz（100ms 間隔），零速時強制立即發送。
 * ========================================================================
 */

// ======================== 設定常數 ========================
const CONFIG = {
    WS_PORT: 8890,                // WebSocket 連接埠（對應後端 common_types.hpp）
    DISPLAY_RANGE_FRONT: 5.0,     // 前方顯示範圍（公尺）
    DISPLAY_RANGE_BACK: 3.0,      // 後方顯示範圍（公尺）
    TARGET_RADIUS: 0.3,           // 目標半徑（公尺）
    MAX_LINEAR_SPEED: 1.0,        // 最大線速度（m/s）
    MAX_ANGULAR_SPEED: 2.0,       // 最大角速度（rad/s）
    RECONNECT_DELAY: 2000,        // 斷線重連延遲（毫秒）

    // 地圖座標校正：3f_routing.png 像素 ↔ 世界座標
    // worldLeft/Right/Bottom/Top = 圖片四邊對應的世界座標
    MAP: {
        worldLeft: -30.0,
        worldRight: 30.0,
        worldBottom: -45.0,
        worldTop: -5.0,
        imgWidth: 1883,
        imgHeight: 735,
        nodeRadius: 18,        // 節點 marker 半徑（像素）
    }
};

// ======================== 全域狀態 ========================
let ws = null;                    // WebSocket 連線物件
let canvas, ctx;                  // LiDAR 視覺化 Canvas 及其 2D 繪圖上下文
let canvasWidth = 800;            // Canvas 寬度（像素，會動態調整）
let canvasHeight = 800;           // Canvas 高度（像素，會動態調整）
let metersToPixels = 100;         // 每公尺對應的像素數
// 機器人在 Canvas 上的中心位置（底部偏上）
let robotCenter = { x: canvasWidth / 2, y: canvasHeight - CONFIG.DISPLAY_RANGE_BACK * metersToPixels };

// ======================== 當前資料 ========================
// 由 WebSocket 接收並更新的雷達及目標資訊
let lidarData = {
    points: [],                       // LiDAR 點雲陣列 [{x, y}, ...]
    target: { x: 0.4, y: 0 },        // 跟隨目標位置（機器人座標系）
    isMovingEnabled: false,           // 運動使能旗標
    isActive: false,                  // 跟隨啟用旗標
    velocity: { vx: 0, vy: 0, wz: 0 }, // 目前速度指令
    rectangleWidth: 0.35              // 目標通道寬度（公尺）
};

// ======================== 導航狀態 ========================
let navState = {
    routeNodes: [],          // 從後端收到的拓撲節點列表 [{name, x, y}, ...]
    navMode: 'route',        // 'route' = 路徑導航, 'goal' = 單點導航
    selectedOrigin: null,    // 已選起點 (node name)
    selectedDest: null,      // 已選終點 (node name)
    selectionStep: 0,        // 0=等起點, 1=等終點
    goalMarker: null,        // 單點導航目標 {x, y} (地圖原始像素)
};

// ======================== Toast 通知 ========================
let toastTimer = null;
function showToast(msg) {
    let el = document.getElementById('navToast');
    if (!el) {
        el = document.createElement('div');
        el.id = 'navToast';
        el.style.cssText = 'position:fixed;top:60px;left:50%;transform:translateX(-50%);' +
            'background:rgba(0,0,0,0.85);color:#e2e8f0;padding:8px 18px;border-radius:20px;' +
            'font-size:0.85rem;z-index:100;pointer-events:none;transition:opacity 0.3s;white-space:nowrap;';
        document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.opacity = '1';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.style.opacity = '0'; }, 2500);
}

// ======================== 地圖檢視器（Pan / Zoom） ========================
const mapView = {
    tx: 0, ty: 0, scale: 1,         // transform 狀態
    dragging: false,                 // 是否正在拖曳
    lastX: 0, lastY: 0,             // 上次指標位置
    pinchDist: 0,                    // 雙指距離（pinch zoom）
    MIN_SCALE: 0.5, MAX_SCALE: 5.0, // 縮放範圍
    el: null, transformEl: null,     // DOM 元素快取

    init() {
        this.el = document.getElementById('mapViewer');
        this.transformEl = document.getElementById('mapTransform');
        if (!this.el || !this.transformEl) return;

        // 初始置中 + fit
        this.resetView();

        // 滑鼠事件
        this.el.addEventListener('mousedown', e => this._onPointerDown(e.clientX, e.clientY, e));
        this.el.addEventListener('mousemove', e => this._onPointerMove(e.clientX, e.clientY));
        this.el.addEventListener('mouseup', () => this._onPointerUp());
        this.el.addEventListener('mouseleave', () => this._onPointerUp());
        this.el.addEventListener('wheel', e => { e.preventDefault(); this._onWheel(e); }, { passive: false });

        // 觸控事件
        this.el.addEventListener('touchstart', e => this._onTouchStart(e), { passive: false });
        this.el.addEventListener('touchmove', e => this._onTouchMove(e), { passive: false });
        this.el.addEventListener('touchend', e => this._onTouchEnd(e));
        this.el.addEventListener('touchcancel', e => this._onTouchEnd(e));

        // 縮放按鈕
        document.getElementById('zoomIn').addEventListener('click', () => this.zoomBy(1.3));
        document.getElementById('zoomOut').addEventListener('click', () => this.zoomBy(1 / 1.3));
        document.getElementById('zoomReset').addEventListener('click', () => this.resetView());
    },

    resetView() {
        if (!this.el || !this.transformEl) return;
        const vw = this.el.clientWidth;
        const vh = this.el.clientHeight;
        const iw = CONFIG.MAP.imgWidth;
        const ih = CONFIG.MAP.imgHeight;
        // fit image into viewer with small padding
        const scaleX = (vw - 10) / iw;
        const scaleY = (vh - 10) / ih;
        this.scale = Math.min(scaleX, scaleY, 1.0);
        this.tx = (vw - iw * this.scale) / 2;
        this.ty = (vh - ih * this.scale) / 2;
        this._apply();
    },

    zoomBy(factor) {
        const cx = this.el.clientWidth / 2;
        const cy = this.el.clientHeight / 2;
        const ns = Math.max(this.MIN_SCALE, Math.min(this.MAX_SCALE, this.scale * factor));
        // 以畫面中心為縮放基準
        this.tx = cx - (cx - this.tx) * (ns / this.scale);
        this.ty = cy - (cy - this.ty) * (ns / this.scale);
        this.scale = ns;
        this._apply();
    },

    /** 把 viewer 內的 clientX/Y 轉成地圖原始像素座標（考慮 transform） */
    clientToMapPixel(cx, cy) {
        const rect = this.el.getBoundingClientRect();
        const vx = cx - rect.left;   // viewer 內座標
        const vy = cy - rect.top;
        const mx = (vx - this.tx) / this.scale;
        const my = (vy - this.ty) / this.scale;
        return { x: mx, y: my };
    },

    _apply() {
        this.transformEl.style.transform = `translate(${this.tx}px,${this.ty}px) scale(${this.scale})`;
        this.transformEl.style.transformOrigin = '0 0';
    },

    _onPointerDown(cx, cy, e) {
        // 不攔截節點 marker 的點擊
        if (e && e.target.closest('.map-node')) return;
        this.dragging = true;
        this.lastX = cx;
        this.lastY = cy;
        this.el.style.cursor = 'grabbing';
    },
    _onPointerMove(cx, cy) {
        if (!this.dragging) return;
        this.tx += cx - this.lastX;
        this.ty += cy - this.lastY;
        this.lastX = cx;
        this.lastY = cy;
        this._apply();
    },
    _onPointerUp() {
        this.dragging = false;
        if (this.el) this.el.style.cursor = 'grab';
    },
    _onWheel(e) {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const rect = this.el.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const ns = Math.max(this.MIN_SCALE, Math.min(this.MAX_SCALE, this.scale * factor));
        this.tx = cx - (cx - this.tx) * (ns / this.scale);
        this.ty = cy - (cy - this.ty) * (ns / this.scale);
        this.scale = ns;
        this._apply();
    },
    _onTouchStart(e) {
        if (e.touches.length === 1) {
            // 單指：記錄起點（不立即 drag，等移動距離 > 門檻才 drag）
            this._onPointerDown(e.touches[0].clientX, e.touches[0].clientY, e);
        } else if (e.touches.length === 2) {
            // 雙指 pinch
            this.pinchDist = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY);
        }
    },
    _onTouchMove(e) {
        e.preventDefault();
        if (e.touches.length === 1 && this.dragging) {
            this._onPointerMove(e.touches[0].clientX, e.touches[0].clientY);
        } else if (e.touches.length === 2) {
            const d = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY);
            if (this.pinchDist > 0) {
                const factor = d / this.pinchDist;
                const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                const rect = this.el.getBoundingClientRect();
                const vx = cx - rect.left;
                const vy = cy - rect.top;
                const ns = Math.max(this.MIN_SCALE, Math.min(this.MAX_SCALE, this.scale * factor));
                this.tx = vx - (vx - this.tx) * (ns / this.scale);
                this.ty = vy - (vy - this.ty) * (ns / this.scale);
                this.scale = ns;
                this._apply();
            }
            this.pinchDist = d;
        }
    },
    _onTouchEnd(e) {
        if (e.touches.length < 2) this.pinchDist = 0;
        if (e.touches.length === 0) this._onPointerUp();
    },
};

// ======================== 初始化入口 ========================
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();                // 初始化 Canvas 及事件監聯
    initWebSocket();             // 建立 WebSocket 連線
    initControls();              // 綁定控制按鈕事件
    initNavPanel();              // 初始化導航面板
    requestAnimationFrame(render); // 啟動渲染迴圈
});

// ======================== Canvas 初始化 ========================
function initCanvas() {
    canvas = document.getElementById('lidarCanvas');
    ctx = canvas.getContext('2d');

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // 桌面端：雙擊設定跟隨目標位置
    canvas.addEventListener('dblclick', handleCanvasClick);

    // 行動端：雙觸摸模擬雙擊
    let lastTouchTime = 0;
    canvas.addEventListener('touchend', (e) => {
        const now = Date.now();
        if (now - lastTouchTime < 300) {  // 300ms 內連續觸摸視為雙擊
            handleCanvasTouch(e);
        }
        lastTouchTime = now;
    });

    initTabSwitching();    // 初始化標籤頁切換
    initDirectControl();   // 初始化直接控制元件（搖桿 + 滑桿）
}

// ======================== Canvas 尺寸調整（支援高 DPI） ========================
function resizeCanvas() {
    if (!canvas || !ctx) return;

    const rect = canvas.parentElement.getBoundingClientRect();
    const containerWidth = rect.width;
    const containerHeight = rect.height;

    // 面板隱藏時容器尺寸為 0，跳過調整
    if (containerWidth <= 0 || containerHeight <= 0) return;

    // 計算 Canvas 尺寸：寬度填滿容器，高度額外加 300px 以顯示更多後方範圍
    const width = containerWidth;
    const height = Math.min(containerHeight, width + 300);

    // 高 DPI 螢幕支援（devicePixelRatio）
    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    ctx.scale(dpr, dpr);
    canvasWidth = width;
    canvasHeight = height;

    // 計算比例尺：水平顯示範圍共 4 公尺（左右各 2 公尺）
    const horizontalRange = 4.0;
    metersToPixels = canvasWidth / horizontalRange;

    // 機器人位置：距離 Canvas 底部 DISPLAY_RANGE_BACK 公尺
    robotCenter = {
        x: canvasWidth / 2,
        y: canvasHeight - CONFIG.DISPLAY_RANGE_BACK * metersToPixels
    };
}

// ======================== 標籤頁切換邏輯 ========================
function initTabSwitching() {
    const tabs = document.querySelectorAll('.nav-item');
    const panels = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 先移除所有標籤的啟用狀態
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => {
                p.classList.remove('active');
                p.style.display = 'none';
            });

            // 啟用當前點擊的標籤
            tab.classList.add('active');
            const targetId = `panel-${tab.dataset.tab}`;
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.style.display = 'flex';
                // 強制重繪以觸發 CSS 淡入動畫
                void targetPanel.offsetWidth;
                targetPanel.classList.add('active');

                // 切換到雷達跟隨面板時，延遲刷新 Canvas 尺寸（等 DOM 更新）
                if (tab.dataset.tab === 'follow') {
                    setTimeout(resizeCanvas, 50);
                }
            }

            // 根據標籤頁發送模式切換指令到後端
            // mode 0: 直接控制, mode 1: 雷達跟隨, mode 2: 導航
            let mode = 1; // 預設跟隨
            if (tab.dataset.tab === 'direct') mode = 0;
            else if (tab.dataset.tab === 'nav') {
                mode = 2;
                // 切到導航面板時重新請求節點 + fit 地圖
                sendCommand({ type: 'get_route_nodes' });
                setTimeout(() => mapView.resetView(), 100);
            }

            sendCommand({
                type: 'switch_mode',
                mode: mode
            });

            // 跟隨 tab：自動啟用追蹤 + 運動 + 切底盤到 Web 模式
            if (tab.dataset.tab === 'follow') {
                sendCommand({ type: 'set_active', active: true });
                sendCommand({ type: 'set_moving', enabled: true });
                sendCommand({ type: 'set_mux_mode', mode: 3 });
            } else {
                sendCommand({ type: 'set_active', active: false });
            }

            // 如果切換到直接控制，可選擇自動開啟運動使能
            /*
            if (mode === 0) {
                sendCommand({ type: 'set_moving', enabled: true });
            }
            */
        });
    });
}

// ======================== 直接控制初始化 ========================
function initDirectControl() {
    initJoystick();          // 虛擬搖桿（控制 vx）
    initSlider();            // 旋轉滑桿（控制 wz）
    initSpeedModeSwitch();   // 速度模式切換（低速/高速）
}

// ======================== 速度模式切換 ========================
function initSpeedModeSwitch() {
    const toggle = document.getElementById('speedModeToggle');
    const container = document.querySelector('.speed-mode-switch');

    if (!toggle || !container) return;

    toggle.addEventListener('change', () => {
        isHighSpeedMode = toggle.checked;
        container.classList.toggle('high-speed', isHighSpeedMode);
        console.log('速度模式:', isHighSpeedMode ? '高速' : '低速');
    });
}

// ======================== 虛擬搖桿邏輯 ========================
// 差速機器人 360° 全向搖桿：
//   Y 軸 → vx（前進後退）
//   X 軸 → wz（轉向，往左前=前進+左轉，像實體手把搖桿）
// 最終 wz = 滑桿 currentWz + 搖桿 joystickWz（疊加），讓滑桿與搖桿可同時微調
let joystickWz = 0;  // 搖桿 X 軸貢獻的角速度

// 合成最終角速度：滑桿 + 搖桿，限制在動作上限內
function combinedWz() {
    const wz = currentWz + joystickWz;
    return Math.max(-CONFIG.MAX_ANGULAR_SPEED, Math.min(CONFIG.MAX_ANGULAR_SPEED, wz));
}

function initJoystick() {
    const base = document.getElementById('joystickBase');
    const stick = document.getElementById('joystickStick');
    const area = document.querySelector('.joystick-area');

    let activeTouchId = null; // 追蹤當前觸控點 ID，避免多點觸控衝突
    const maxRadius = 140;    // 搖桿最大位移半徑（像素），適配 360px 底盤

    // 觸控/滑鼠按下：記錄觸控 ID 並更新搖桿位置
    function handleStart(e) {
        if (e.type === 'mousedown') {
            activeTouchId = 'mouse';
        } else {
            activeTouchId = e.changedTouches[0].identifier;
        }
        const touch = e.type === 'mousedown' ? e : e.changedTouches[0];
        const rect = base.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        updateJoystick(touch.clientX, touch.clientY, centerX, centerY);
    }

    // 觸控/滑鼠移動：持續更新搖桿位置與速度
    function handleMove(e) {
        if (activeTouchId === null) return;
        e.preventDefault();

        let touch;
        if (e.type === 'mousemove') {
            if (activeTouchId !== 'mouse') return;
            touch = e;
        } else {
            // 從所有觸控點中找到對應 ID 的那個
            touch = Array.from(e.touches).find(t => t.identifier === activeTouchId);
            if (!touch) return;
        }

        const rect = base.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        updateJoystick(touch.clientX, touch.clientY, centerX, centerY);
    }

    // 觸控/滑鼠放開：搖桿回中，vx 歸零（保持當前角速度 wz）
    function handleEnd(e) {
        if (activeTouchId === null) return;

        if (e.type === 'mouseup') {
            if (activeTouchId !== 'mouse') return;
        } else {
            const ended = Array.from(e.changedTouches).find(t => t.identifier === activeTouchId);
            if (!ended) return;
        }

        activeTouchId = null;
        stick.style.transform = `translate(-50%, -50%)`;
        // 搖桿回中：vx 與搖桿轉向都歸零，保留滑桿的 currentWz
        joystickWz = 0;
        sendVelocity(0, 0, combinedWz());
    }

    // 計算搖桿位移並轉換為速度指令
    function updateJoystick(clientX, clientY, centerX, centerY) {
        let dx = clientX - centerX;
        let dy = clientY - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // 限制搖桿位移不超過最大半徑
        if (distance > maxRadius) {
            const angle = Math.atan2(dy, dx);
            dx = Math.cos(angle) * maxRadius;
            dy = Math.sin(angle) * maxRadius;
        }

        stick.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;

        // 360° 全向搖桿（差速機器人）:
        //   Y 軸（上下）→ vx 前進後退（向上為正）
        //   X 軸（左右）→ joystickWz 轉向（向左 dx<0 → wz>0 左轉）
        const vx = -(dy / maxRadius) * CONFIG.MAX_LINEAR_SPEED;
        const vy = 0;
        joystickWz = -(dx / maxRadius) * CONFIG.MAX_ANGULAR_SPEED;

        sendVelocity(vx, vy, combinedWz());
    }

    // 桌面端事件綁定
    base.addEventListener('mousedown', handleStart);
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleEnd);

    // 行動端事件綁定（passive: false 以允許 preventDefault）
    base.addEventListener('touchstart', handleStart, { passive: false });
    document.addEventListener('touchmove', handleMove, { passive: false });
    document.addEventListener('touchend', handleEnd);
    document.addEventListener('touchcancel', handleEnd);
}

// ======================== 旋轉滑桿邏輯 ========================
// 水平滑桿控制角速度 wz：向左 → wz > 0（逆時針），向右 → wz < 0（順時針）
let currentWz = 0;
function initSlider() {
    const knob = document.getElementById('rotationKnob');
    const container = document.querySelector('.slider-container');

    let activeTouchId = null; // 追蹤觸控點 ID

    // 按下旋鈕：記錄觸控 ID
    function handleStart(e) {
        if (e.type === 'mousedown') {
            activeTouchId = 'mouse';
        } else {
            activeTouchId = e.changedTouches[0].identifier;
        }
    }

    // 拖曳旋鈕：計算水平偏移量並映射為角速度
    function handleMove(e) {
        if (activeTouchId === null) return;
        e.preventDefault();

        let touch;
        if (e.type === 'mousemove') {
            if (activeTouchId !== 'mouse') return;
            touch = e;
        } else {
            touch = Array.from(e.touches).find(t => t.identifier === activeTouchId);
            if (!touch) return;
        }

        const rect = container.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;

        // 計算偏移量並限制在軌道範圍內
        let offset = touch.clientX - centerX;
        offset = Math.max(-rect.width / 2 + 20, Math.min(rect.width / 2 - 20, offset));

        knob.style.transform = `translateX(calc(-50% + ${offset}px))`;

        // 映射角速度: 向左(offset<0) → wz>0(逆時針), 向右(offset>0) → wz<0(順時針)
        currentWz = -(offset / (rect.width / 2)) * CONFIG.MAX_ANGULAR_SPEED;
        sendVelocity(lastVx, lastVy, combinedWz());
    }

    // 放開旋鈕：回中歸零
    function handleEnd(e) {
        if (activeTouchId === null) return;

        if (e.type === 'mouseup') {
            if (activeTouchId !== 'mouse') return;
        } else {
            const ended = Array.from(e.changedTouches).find(t => t.identifier === activeTouchId);
            if (!ended) return;
        }

        activeTouchId = null;
        knob.style.transform = `translateX(-50%)`;
        currentWz = 0;
        sendVelocity(lastVx, lastVy, combinedWz());
    }

    // 桌面端事件綁定
    knob.addEventListener('mousedown', handleStart);
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleEnd);

    // 行動端事件綁定
    knob.addEventListener('touchstart', handleStart, { passive: false });
    document.addEventListener('touchmove', handleMove, { passive: false });
    document.addEventListener('touchend', handleEnd);
    document.addEventListener('touchcancel', handleEnd);
}

// ======================== 速度發送邏輯 ========================

// 快取最後的速度指令（搖桿放開時需保留角速度）
let lastVx = 0;
let lastVy = 0;

// 速度模式：false = 低速（0.5 倍），true = 高速（1.0 倍）
let isHighSpeedMode = false;
const LOW_SPEED_RATIO = 0.5;
const HIGH_SPEED_RATIO = 1.0;

// 發送限頻：10Hz（100ms 間隔），避免 WebSocket 過載
let lastSendTime = 0;
const MIN_SEND_INTERVAL = 100;

// 暫存待發送的速度值
let pendingVx = 0;
let pendingVy = 0;
let pendingWz = 0;
let velocitySendTimer = null;

/**
 * 速度發送入口：快取最新速度，依限頻規則決定是否立即發送
 * 零速指令強制立即發送（確保機器人能即時停止）
 */
function sendVelocity(vx, vy, wz) {
    lastVx = vx;
    lastVy = vy;

    // 快取最新速度
    pendingVx = vx;
    pendingVy = vy;
    pendingWz = wz;

    // 零速時強制立即發送，否則依 10Hz 限頻
    const isZeroVelocity = (vx === 0 && vy === 0 && wz === 0);
    const now = Date.now();

    if (isZeroVelocity || now - lastSendTime >= MIN_SEND_INTERVAL) {
        doSendVelocity();
    }
}

/**
 * 實際發送速度指令：套用速度倍率 + 死區過濾後，透過 WebSocket 送出
 */
function doSendVelocity() {
    lastSendTime = Date.now();

    // 根據速度模式套用倍率
    const ratio = isHighSpeedMode ? HIGH_SPEED_RATIO : LOW_SPEED_RATIO;
    let scaledVx = pendingVx * ratio;
    let scaledVy = pendingVy * ratio;
    let scaledWz = pendingWz * ratio;

    // 死區過濾：微小值歸零，避免底層控制器對複合指令回應不佳
    if (Math.abs(scaledVx) < 0.05) scaledVx = 0;
    if (Math.abs(scaledVy) < 0.05) scaledVy = 0;
    if (Math.abs(scaledWz) < 0.05) scaledWz = 0;

    console.log(`[${Date.now()}] 發送速度: vx=${scaledVx.toFixed(2)}, vy=${scaledVy.toFixed(2)}, wz=${scaledWz.toFixed(2)}`);

    // 送出 direct_cmd 類型的 JSON 指令
    sendCommand({
        type: 'direct_cmd',
        x: scaledVx,
        y: scaledVy,
        z: scaledWz
    });
}

// ======================== 座標轉換 ========================

/**
 * 機器人座標 → Canvas 像素座標
 * 機器人座標系：x 朝前、y 朝左
 * Canvas 座標系：x 朝右、y 朝下
 */
function toPixel(robotX, robotY) {
    return {
        x: robotCenter.x - robotY * metersToPixels,   // 機器人 y 朝左 → Canvas x 朝右取負
        y: robotCenter.y - robotX * metersToPixels     // 機器人 x 朝前 → Canvas y 朝上取負
    };
}

/**
 * Canvas 像素座標 → 機器人座標（toPixel 的反函數）
 */
function toRobot(pixelX, pixelY) {
    return {
        x: (robotCenter.y - pixelY) / metersToPixels,
        y: (robotCenter.x - pixelX) / metersToPixels
    };
}

// ======================== Canvas 互動事件 ========================

// 桌面端雙擊：在 Canvas 上設定跟隨目標
function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setTarget(x, y);
}

// 行動端雙觸摸：在 Canvas 上設定跟隨目標
function handleCanvasTouch(e) {
    e.preventDefault();
    if (e.changedTouches.length > 0) {
        const touch = e.changedTouches[0];
        const rect = canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        setTarget(x, y);
    }
}

/**
 * 設定跟隨目標位置：將像素座標轉換為機器人座標後發送給後端
 */
function setTarget(pixelX, pixelY) {
    const robot = toRobot(pixelX, pixelY);
    sendCommand({
        type: 'set_target',
        x: robot.x,
        y: robot.y
    });
}

// ======================== WebSocket 連線管理 ========================

/**
 * 初始化 WebSocket 連線
 * 自動偵測目前頁面的 hostname，連接到 CONFIG.WS_PORT（8890）
 * 斷線後自動重連（延遲 RECONNECT_DELAY 毫秒）
 */
function initWebSocket() {
    const host = window.location.hostname || 'localhost';
    const wsUrl = `ws://${host}:${CONFIG.WS_PORT}`;

    updateConnectionStatus('connecting');

    try {
        ws = new WebSocket(wsUrl);

        // 連線成功：發送目前所在標籤頁對應的控制模式
        ws.onopen = () => {
            console.log('WebSocket 已連線');
            updateConnectionStatus('connected');

            const activeTab = document.querySelector('.nav-item.active');
            if (activeTab) {
                let mode = 1; // 預設跟隨
                if (activeTab.dataset.tab === 'direct') mode = 0;
                else if (activeTab.dataset.tab === 'nav') mode = 2;

                sendCommand({
                    type: 'switch_mode',
                    mode: mode
                });
                console.log('發送初始控制模式:', mode);

                // 請求路由節點列表
                sendCommand({ type: 'get_route_nodes' });
            }
        };

        // 收到訊息：解析 JSON 並更新資料
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleData(data);
            } catch (e) {
                console.error('解析資料失敗:', e);
            }
        };

        // 連線斷開：更新狀態並排程重連
        ws.onclose = () => {
            console.log('WebSocket 已斷開');
            updateConnectionStatus('disconnected');
            setTimeout(initWebSocket, CONFIG.RECONNECT_DELAY);
        };

        // 連線錯誤
        ws.onerror = (error) => {
            console.error('WebSocket 錯誤:', error);
            updateConnectionStatus('disconnected');
        };
    } catch (e) {
        console.error('建立 WebSocket 失敗:', e);
        updateConnectionStatus('disconnected');
        setTimeout(initWebSocket, CONFIG.RECONNECT_DELAY);
    }
}

// ======================== 連線狀態 UI 更新 ========================

/**
 * 更新導覽列的連線狀態指示器
 * @param {string} status - 'connecting' | 'connected' | 'disconnected'
 */
function updateConnectionStatus(status) {
    const el = document.getElementById('connectionStatus');
    const textEl = el.querySelector('.status-text');

    el.className = 'connection-status ' + status;

    switch (status) {
        case 'connected':
            textEl.textContent = '已連接';
            break;
        case 'disconnected':
            textEl.textContent = '已斷開';
            break;
        default:
            textEl.textContent = '連接中...';
    }
}

// ======================== 資料處理 ========================

/**
 * 處理從 WebSocket 接收的資料
 * 目前只處理 'scan_data' 類型：包含 LiDAR 點雲、目標位置、速度指令等
 */
function handleData(data) {
    if (data.type === 'scan_data') {
        lidarData.points = data.points || [];                // LiDAR 點雲
        lidarData.target = data.target || lidarData.target;  // 跟隨目標座標
        lidarData.isMovingEnabled = data.is_moving_enabled || false; // 運動使能
        lidarData.isActive = data.is_active || false;        // 跟隨啟用
        lidarData.velocity = {
            vx: data.velocity?.vx || 0,  // 前後線速度
            vy: data.velocity?.vy || 0,  // 橫向線速度（差速機器人應為 0）
            wz: data.velocity?.wz || 0   // 角速度
        };
        lidarData.rectangleWidth = data.rectangle_width || 0.35; // 目標通道寬度

        updateUI();
    } else if (data.type === 'route_nodes') {
        // 收到路由節點列表 → 更新導航狀態 + 重新繪製節點
        navState.routeNodes = data.nodes || [];
        renderMapNodes();
        console.log('收到路由節點:', navState.routeNodes.length);
    }
}

// ======================== UI 更新 ========================

/**
 * 根據最新的 lidarData 更新右側面板的 UI 元件
 */
function updateUI() {
    // 更新目標位置顯示
    document.getElementById('targetX').textContent = lidarData.target.x.toFixed(2) + ' m';
    document.getElementById('targetY').textContent = lidarData.target.y.toFixed(2) + ' m';

    // 繪製速度顯示圖表
    drawSpeedDisplay();

    // 更新運動使能按鈕狀態（運動中顯示「停止」，停止時顯示「開始」）
    updateButtonState('movingToggle', lidarData.isMovingEnabled, '停止運動', '開始運動', '⏸️', '▶️');
}

// ======================== 速度顯示圖表 ========================

/**
 * 在 speedCanvas 上繪製速度視覺化：
 * - 十字座標顯示 vx（紅色/垂直）和 vy（綠色/水平）
 * - 圓弧顯示 wz（藍色）
 * - 右側標籤顯示數值
 */
function drawSpeedDisplay() {
    const canvas = document.getElementById('speedCanvas');
    if (!canvas) return;

    // 自適應容器寬度
    const container = canvas.parentElement;
    if (container) {
        canvas.width = container.clientWidth || 180;
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 清除畫布（深色背景）
    ctx.fillStyle = '#1a2332';
    ctx.fillRect(0, 0, width, height);

    // 根據寬度計算佈局參數
    const barLength = Math.min(30, height / 2 - 10);  // 速度條最大長度
    const arcRadius = barLength + 6;                    // 角速度圓弧半徑
    const centerX = arcRadius + 5;                      // 圓弧中心 X（左邊距）
    const centerY = height / 2;                         // 圓弧中心 Y（垂直置中）
    const labelX = centerX + arcRadius + 8;             // 數值標籤起始 X

    const vx = lidarData.velocity.vx;
    const vy = lidarData.velocity.vy;
    const wz = lidarData.velocity.wz;

    // --- 繪製角速度 wz 圓弧背景 ---
    ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, arcRadius, 0, Math.PI * 2);
    ctx.stroke();

    // 圓弧零點標記（頂部小線段）
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - arcRadius - 4);
    ctx.lineTo(centerX, centerY - arcRadius + 4);
    ctx.stroke();

    // --- 繪製角速度 wz 弧線（藍色） ---
    if (Math.abs(wz) > 0.01) {
        const startAngle = -Math.PI / 2;  // 從頂部開始
        const arcAngle = -(wz / CONFIG.MAX_ANGULAR_SPEED) * Math.PI;
        const endAngle = startAngle + arcAngle;

        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        if (arcAngle > 0) {
            ctx.arc(centerX, centerY, arcRadius, startAngle, endAngle);
        } else {
            ctx.arc(centerX, centerY, arcRadius, endAngle, startAngle);
        }
        ctx.stroke();

        // 弧線末端圓點
        const arrowAngle = endAngle;
        const arrowX = centerX + arcRadius * Math.cos(arrowAngle);
        const arrowY = centerY + arcRadius * Math.sin(arrowAngle);
        ctx.beginPath();
        ctx.arc(arrowX, arrowY, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
    }

    // --- 繪製十字參考線 ---
    ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - barLength - 3, centerY);
    ctx.lineTo(centerX + barLength + 3, centerY);
    ctx.moveTo(centerX, centerY - barLength - 3);
    ctx.lineTo(centerX, centerY + barLength + 3);
    ctx.stroke();

    // 計算速度條長度（依比例縮放）
    const barX = (vx / CONFIG.MAX_LINEAR_SPEED) * barLength;
    const barY = (vy / CONFIG.MAX_LINEAR_SPEED) * barLength;

    // --- 繪製 vx 速度條（紅色/垂直，向上為正） ---
    if (Math.abs(barX) > 1) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX, centerY - barX);
        ctx.stroke();

        // 箭頭
        const arrowY = centerY - barX;
        const arrowDir = barX > 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(centerX, arrowY - arrowDir * 5);
        ctx.lineTo(centerX - 3, arrowY);
        ctx.lineTo(centerX + 3, arrowY);
        ctx.closePath();
        ctx.fillStyle = '#ef4444';
        ctx.fill();
    }

    // --- 繪製 vy 速度條（綠色/水平，向左為正） ---
    if (Math.abs(barY) > 1) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX - barY, centerY);
        ctx.stroke();

        // 箭頭
        const arrowX = centerX - barY;
        const arrowDir = barY > 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(arrowX + arrowDir * 5, centerY);
        ctx.lineTo(arrowX, centerY - 3);
        ctx.lineTo(arrowX, centerY + 3);
        ctx.closePath();
        ctx.fillStyle = '#10b981';
        ctx.fill();
    }

    // 中心點（白色）
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // --- 右側速度數值標籤 ---
    ctx.font = '16px sans-serif';
    const valueX = width - 5; // 右邊距 5px

    // Vx（紅色）
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ef4444';
    ctx.fillText('Vx:', labelX, centerY - 18);
    ctx.textAlign = 'right';
    ctx.fillText(vx.toFixed(2), valueX, centerY - 18);

    // Vy（綠色）
    ctx.textAlign = 'left';
    ctx.fillStyle = '#10b981';
    ctx.fillText('Vy:', labelX, centerY + 2);
    ctx.textAlign = 'right';
    ctx.fillText(vy.toFixed(2), valueX, centerY + 2);

    // Wz（藍色）
    ctx.textAlign = 'left';
    ctx.fillStyle = '#3b82f6';
    ctx.fillText('Wz:', labelX, centerY + 22);
    ctx.textAlign = 'right';
    ctx.fillText(wz.toFixed(2), valueX, centerY + 22);
}

// ======================== 按鈕狀態更新 ========================

/**
 * 更新按鈕的文字、圖示及啟用狀態 CSS class
 */
function updateButtonState(id, isActive, activeText, inactiveText, activeIcon, inactiveIcon) {
    const btn = document.getElementById(id);
    btn.classList.toggle('active', isActive);
    btn.querySelector('.btn-text').textContent = isActive ? activeText : inactiveText;
    btn.querySelector('.btn-icon').textContent = isActive ? activeIcon : inactiveIcon;
}

// ======================== 控制按鈕事件綁定 ========================

/**
 * 初始化運動使能按鈕的點擊事件
 * 差速機器人無需趴下/站起動作按鈕
 */
function initControls() {
    document.getElementById('movingToggle').addEventListener('click', () => {
        sendCommand({
            type: 'set_moving',
            enabled: !lidarData.isMovingEnabled
        });
    });

    // 底盤模式切換按鈕
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = parseInt(btn.dataset.mode);
            sendCommand({ type: 'set_mux_mode', mode: mode });
            // 更新 UI：移除其他按鈕的 active，加上自己的
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

// ======================== WebSocket 指令發送 ========================

/**
 * 透過 WebSocket 發送 JSON 指令到後端
 * @param {Object} cmd - 指令物件，例如 { type: 'direct_cmd', x, y, z }
 */
function sendCommand(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(cmd));
    }
}

// ======================== 渲染迴圈 ========================

/**
 * requestAnimationFrame 渲染迴圈：每幀重繪 LiDAR 視覺化
 */
function render() {
    drawLidar();
    requestAnimationFrame(render);
}

// ======================== LiDAR 視覺化繪製 ========================

/**
 * 繪製完整的 LiDAR 雷達圖：網格 → 目標區域 → 點雲 → 目標 → 機器人 → 狀態文字
 */
function drawLidar() {
    const dpr = window.devicePixelRatio || 1;
    ctx.resetTransform();
    ctx.scale(dpr, dpr);

    // 清除畫布（深色背景）
    ctx.fillStyle = '#0a0e17';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    drawGrid();        // 同心圓距離網格
    drawTargetZone();  // 目標連接線與矩形通道
    drawPoints();      // LiDAR 點雲
    drawTarget();      // 目標位置圓圈
    drawRobot();       // 機器人圖示
    drawStatus();      // 左上角運動狀態文字
}

// ======================== 網格繪製 ========================

/**
 * 繪製同心圓距離環（1m~4m）和十字參考線
 */
function drawGrid() {
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.1)';
    ctx.lineWidth = 1;

    // 同心圓（每 1 公尺一圈）
    for (let r = 1; r <= 4; r++) {
        ctx.beginPath();
        ctx.arc(robotCenter.x, robotCenter.y, r * metersToPixels, 0, Math.PI * 2);
        ctx.stroke();

        // 距離標籤
        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '12px sans-serif';
        ctx.fillText(r + 'm', robotCenter.x + 5, robotCenter.y - r * metersToPixels + 15);
    }

    // 十字參考線（水平 + 垂直）
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.15)';
    ctx.beginPath();
    ctx.moveTo(0, robotCenter.y);
    ctx.lineTo(canvasWidth, robotCenter.y);
    ctx.moveTo(robotCenter.x, 0);
    ctx.lineTo(robotCenter.x, canvasHeight);
    ctx.stroke();
}

// ======================== 目標區域繪製 ========================

/**
 * 繪製從機器人到目標的矩形通道及虛線連接線
 * 矩形通道用於判斷障礙物是否阻擋路徑
 */
function drawTargetZone() {
    const target = lidarData.target;
    const dist = Math.sqrt(target.x * target.x + target.y * target.y);
    const rectWidth = lidarData.rectangleWidth;

    if (dist > 0.01) {
        const targetPixel = toPixel(target.x, target.y);
        const angle = Math.atan2(target.y, target.x);
        const halfWidth = rectWidth / 2.0;

        // 計算矩形四個角落（機器人座標系），與後端演算法一致
        const corners = [
            { x: 0 - halfWidth * Math.sin(angle), y: 0 + halfWidth * Math.cos(angle) },
            { x: 0 + halfWidth * Math.sin(angle), y: 0 - halfWidth * Math.cos(angle) },
            { x: target.x + halfWidth * Math.sin(angle), y: target.y - halfWidth * Math.cos(angle) },
            { x: target.x - halfWidth * Math.sin(angle), y: target.y + halfWidth * Math.cos(angle) }
        ];

        // 轉換為像素座標
        const pixelCorners = corners.map(c => toPixel(c.x, c.y));

        // 填充半透明矩形
        ctx.fillStyle = 'rgba(80, 80, 80, 0.5)';
        ctx.beginPath();
        ctx.moveTo(pixelCorners[0].x, pixelCorners[0].y);
        ctx.lineTo(pixelCorners[1].x, pixelCorners[1].y);
        ctx.lineTo(pixelCorners[2].x, pixelCorners[2].y);
        ctx.lineTo(pixelCorners[3].x, pixelCorners[3].y);
        ctx.closePath();
        ctx.fill();

        // 矩形邊框
        ctx.strokeStyle = 'rgba(100, 100, 100, 0.8)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // 機器人到目標的虛線連接線（綠色）
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(robotCenter.x, robotCenter.y);
        ctx.lineTo(targetPixel.x, targetPixel.y);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// ======================== LiDAR 點雲繪製 ========================

/**
 * 繪製 LiDAR 點雲：
 * - 普通點（紅色）：不在目標通道內的點
 * - 特殊點（黃色）：在目標通道內的障礙物點，額外繪製到機器人的連線
 */
function drawPoints() {
    const target = lidarData.target;
    const targetSq = target.x * target.x + target.y * target.y;
    const targetLen = Math.sqrt(targetSq);
    const rectHalfWidth = lidarData.rectangleWidth / 2.0;

    const normalPoints = [];   // 普通點（通道外）
    const specialPoints = [];  // 特殊點（通道內障礙物）

    for (const point of lidarData.points) {
        let isSpecial = false;

        if (targetLen > 0.01) {
            // 將點投影到目標方向：計算沿目標方向的投影長度
            const proj_x = (point.x * target.x + point.y * target.y) / targetLen;

            // 計算垂直於目標方向的距離（用叉積）
            const proj_y = (point.y * target.x - point.x * target.y) / targetLen;

            // 判斷是否在矩形通道內（投影在 0~targetLen 之間，且垂直距離在半寬內）
            if (proj_x >= 0 && proj_x <= targetLen && Math.abs(proj_y) <= rectHalfWidth) {
                // 排除目標圓內的點（目標本身不算障礙物）
                const dx = point.x - target.x;
                const dy = point.y - target.y;
                if (dx * dx + dy * dy > CONFIG.TARGET_RADIUS * CONFIG.TARGET_RADIUS) {
                    isSpecial = true;
                }
            }
        }

        // 轉換為像素座標
        const p = toPixel(point.x, point.y);

        if (isSpecial) {
            specialPoints.push(p);
        } else {
            normalPoints.push(p);
        }
    }

    // 1. 繪製特殊點的連線（半透明黃色，從機器人到障礙物點）
    if (specialPoints.length > 0) {
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.3)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (const p of specialPoints) {
            ctx.moveTo(robotCenter.x, robotCenter.y);
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
    }

    // 2. 繪製普通點（紅色小圓點）
    if (normalPoints.length > 0) {
        ctx.fillStyle = '#ef4444';
        ctx.shadowColor = '#ef4444';
        ctx.shadowBlur = 3;
        ctx.beginPath();
        for (const p of normalPoints) {
            ctx.moveTo(p.x + 2, p.y);
            ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        }
        ctx.fill();
    }

    // 3. 繪製特殊點（黃色較大圓點，強調障礙物）
    if (specialPoints.length > 0) {
        ctx.fillStyle = '#ffff00';
        ctx.shadowColor = '#ffff00';
        ctx.shadowBlur = 5;
        ctx.beginPath();
        for (const p of specialPoints) {
            ctx.moveTo(p.x + 3, p.y);
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        }
        ctx.fill();
    }

    ctx.shadowBlur = 0;
}

// ======================== 目標繪製 ========================

/**
 * 繪製跟隨目標：紫色圓圈 + 十字準星
 */
function drawTarget() {
    const target = lidarData.target;
    const p = toPixel(target.x, target.y);
    const radius = CONFIG.TARGET_RADIUS * metersToPixels;

    // 目標圓（紫色發光效果）
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#a855f7';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.stroke();

    // 十字準星
    ctx.beginPath();
    ctx.moveTo(p.x - 10, p.y);
    ctx.lineTo(p.x + 10, p.y);
    ctx.moveTo(p.x, p.y - 10);
    ctx.lineTo(p.x, p.y + 10);
    ctx.stroke();

    ctx.shadowBlur = 0;
}

// ======================== 機器人繪製 ========================

/**
 * 繪製機器人本體：青色圓點 + 朝前方向箭頭
 */
function drawRobot() {
    // 機器人主體（青色發光圓點）
    ctx.fillStyle = '#22d3ee';
    ctx.shadowColor = '#22d3ee';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(robotCenter.x, robotCenter.y, 12, 0, Math.PI * 2);
    ctx.fill();

    // 方向指示線（朝上 = 朝前）
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(robotCenter.x, robotCenter.y);
    ctx.lineTo(robotCenter.x, robotCenter.y - 25);
    ctx.stroke();

    // 方向箭頭
    ctx.beginPath();
    ctx.moveTo(robotCenter.x, robotCenter.y - 30);
    ctx.lineTo(robotCenter.x - 8, robotCenter.y - 20);
    ctx.lineTo(robotCenter.x + 8, robotCenter.y - 20);
    ctx.closePath();
    ctx.fill();

    ctx.shadowBlur = 0;
}

// ======================== 狀態文字繪製 ========================

/**
 * 在 Canvas 左上角繪製運動狀態文字（MOVING / STOPPED）
 */
function drawStatus() {
    const status = lidarData.isMovingEnabled ? 'MOVING' : 'STOPPED';
    const color = lidarData.isMovingEnabled ? '#10b981' : '#ef4444';

    ctx.font = 'bold 16px sans-serif';
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 5;
    ctx.fillText(status, 10, 22);

    ctx.shadowBlur = 0;
}

// ======================== 導航面板邏輯 ========================

/**
 * 座標轉換：世界座標 → 地圖圖片像素
 */
function worldToMapPixel(wx, wy) {
    const m = CONFIG.MAP;
    const px = ((wx - m.worldLeft) / (m.worldRight - m.worldLeft)) * m.imgWidth;
    const py = ((m.worldTop - wy) / (m.worldTop - m.worldBottom)) * m.imgHeight;
    return { x: px, y: py };
}

/**
 * 座標轉換：地圖圖片像素 → 世界座標
 */
function mapPixelToWorld(px, py) {
    const m = CONFIG.MAP;
    const wx = m.worldLeft + (px / m.imgWidth) * (m.worldRight - m.worldLeft);
    const wy = m.worldTop - (py / m.imgHeight) * (m.worldTop - m.worldBottom);
    return { x: wx, y: wy };
}

/**
 * 初始化導航面板：子模式切換、地圖 viewer、按鈕
 */
function initNavPanel() {
    // 初始化地圖 pan/zoom viewer
    mapView.init();

    // 子模式切換
    document.querySelectorAll('.submode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.submode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            navState.navMode = btn.dataset.navmode;
            navState.selectedOrigin = null;
            navState.selectedDest = null;
            navState.selectionStep = 0;
            updateNavUI();
        });
    });

    // 地圖單擊 tap → 選擇路由節點（用延遲區分單擊/雙擊）
    let pointerDownPos = null;
    let singleTapTimer = null;
    const viewer = document.getElementById('mapViewer');
    viewer.addEventListener('pointerdown', e => {
        if (e.target.closest('.map-node')) return;
        if (e.target.closest('.zoom-btn')) return;
        pointerDownPos = { x: e.clientX, y: e.clientY };
    });
    viewer.addEventListener('pointerup', e => {
        if (!pointerDownPos) return;
        const dx = e.clientX - pointerDownPos.x;
        const dy = e.clientY - pointerDownPos.y;
        pointerDownPos = null;
        // 移動 > 8px 視為 drag，不觸發 tap
        if (Math.hypot(dx, dy) > 8) return;
        // 延遲 300ms 再觸發單擊（雙擊會取消它）
        const cx = e.clientX, cy = e.clientY;
        clearTimeout(singleTapTimer);
        singleTapTimer = setTimeout(() => {
            handleMapTap(cx, cy);
        }, 300);
    });

    // 地圖雙擊/雙 tap → 單點導航（兩種模式都能用）
    let lastTapTime = 0;
    let lastTapPos = { x: 0, y: 0 };

    // 統一用 pointerup 偵測快速連點（滑鼠 + 觸控都走 pointer）
    viewer.addEventListener('pointerup', e => {
        if (e.target.closest('.map-node')) return;
        if (e.target.closest('.zoom-btn')) return;
        const now = Date.now();
        const dx = Math.abs(e.clientX - lastTapPos.x);
        const dy = Math.abs(e.clientY - lastTapPos.y);
        // 400ms 內同位置連擊 = 雙擊
        if (now - lastTapTime < 400 && dx < 30 && dy < 30) {
            clearTimeout(singleTapTimer);  // 取消待發的單擊
            handleMapDblTap(e.clientX, e.clientY);
            lastTapTime = 0;  // 重置避免三連擊
        } else {
            lastTapTime = now;
        }
        lastTapPos = { x: e.clientX, y: e.clientY };
    });

    // 節點 marker 點擊（事件代理）
    document.getElementById('mapOverlay').addEventListener('click', e => {
        const node = e.target.closest('.map-node');
        if (!node) return;
        selectNode(node.dataset.name);
    });

    // 開始導航按鈕
    document.getElementById('startRouteNav').addEventListener('click', () => {
        if (navState.selectedOrigin && navState.selectedDest) {
            sendCommand({
                type: 'route_nav',
                origin: navState.selectedOrigin,
                destination: navState.selectedDest
            });
            console.log(`路徑導航: ${navState.selectedOrigin} → ${navState.selectedDest}`);
            sendCommand({ type: 'set_mux_mode', mode: 3 });
        }
    });

    // 取消按鈕
    document.getElementById('cancelRouteNav').addEventListener('click', () => {
        navState.selectedOrigin = null;
        navState.selectedDest = null;
        navState.selectionStep = 0;
        updateNavUI();
    });
}

/**
 * 單次 tap → 路徑導航選節點
 */
function handleMapTap(cx, cy) {
    if (navState.navMode !== 'route') return;
    if (navState.routeNodes.length === 0) {
        showToast('節點載入中，請稍候…');
        return;
    }

    const mp = mapView.clientToMapPixel(cx, cy);

    // 找最近的節點（地圖原始像素距離）
    let minDist = Infinity, nearest = null;
    for (const node of navState.routeNodes) {
        const np = worldToMapPixel(node.x, node.y);
        const d = Math.hypot(np.x - mp.x, np.y - mp.y);
        if (d < minDist) { minDist = d; nearest = node; }
    }
    // 門檻：地圖原始像素 30px / 當前縮放
    const threshold = 30 / mapView.scale * 2;
    if (!nearest || minDist > Math.max(30, threshold)) {
        showToast('請點選藍色節點');
        return;
    }
    selectNode(nearest.name);
}

/**
 * 節點 marker 點擊或 tap 選到 → 更新選擇狀態
 */
function selectNode(name) {
    if (navState.navMode !== 'route') return;
    if (navState.selectionStep === 0) {
        navState.selectedOrigin = name;
        navState.selectionStep = 1;
        showToast(`🟢 起點: ${name}`);
    } else {
        navState.selectedDest = name;
        navState.selectionStep = 0;
        showToast(`🔴 終點: ${name}`);
    }
    updateNavUI();
    renderMapNodes();
}

/**
 * 雙擊 / 雙 tap → 單點導航（任何模式都能用）
 */
function handleMapDblTap(cx, cy) {
    const mp = mapView.clientToMapPixel(cx, cy);
    // 超出圖片範圍就忽略
    if (mp.x < 0 || mp.x > CONFIG.MAP.imgWidth || mp.y < 0 || mp.y > CONFIG.MAP.imgHeight) return;

    const world = mapPixelToWorld(mp.x, mp.y);
    navState.goalMarker = { x: mp.x, y: mp.y };
    renderGoalMarker();
    showToast(`🎯 目標: (${world.x.toFixed(1)}, ${world.y.toFixed(1)})`);
    sendCommand({ type: 'goal_nav', x: world.x, y: world.y });
    sendCommand({ type: 'set_mux_mode', mode: 3 });
}

/**
 * 在地圖上顯示單點導航目標 marker
 */
function renderGoalMarker() {
    // 移除舊的
    const old = document.getElementById('goalMarker');
    if (old) old.remove();
    if (!navState.goalMarker) return;

    const overlay = document.getElementById('mapOverlay');
    const m = document.createElement('div');
    m.id = 'goalMarker';
    m.style.cssText = `position:absolute;left:${navState.goalMarker.x}px;top:${navState.goalMarker.y}px;` +
        `transform:translate(-50%,-50%);width:30px;height:30px;border-radius:50%;` +
        `background:rgba(239,68,68,0.6);border:3px solid #ef4444;color:#fff;` +
        `display:flex;align-items:center;justify-content:center;font-size:14px;z-index:5;` +
        `box-shadow:0 0 12px rgba(239,68,68,0.6);pointer-events:none;`;
    m.textContent = '🎯';
    overlay.appendChild(m);
}

/**
 * 渲染節點 markers 覆蓋在圖片上（使用原始像素座標）
 */
function renderMapNodes() {
    const overlay = document.getElementById('mapOverlay');
    overlay.innerHTML = '';
    if (navState.routeNodes.length === 0) return;

    for (const node of navState.routeNodes) {
        const pixel = worldToMapPixel(node.x, node.y);
        const marker = document.createElement('div');
        marker.className = 'map-node';
        marker.dataset.name = node.name;
        // 使用絕對像素（overlay 固定 1883×735）
        marker.style.left = `${pixel.x}px`;
        marker.style.top = `${pixel.y}px`;
        marker.textContent = node.name;

        if (node.name === navState.selectedOrigin) marker.classList.add('origin');
        if (node.name === navState.selectedDest) marker.classList.add('dest');

        overlay.appendChild(marker);
    }
}

/**
 * 更新導航面板 UI 狀態
 */
function updateNavUI() {
    const originEl = document.getElementById('selectedOrigin');
    const destEl = document.getElementById('selectedDest');
    const startBtn = document.getElementById('startRouteNav');
    const hint = document.getElementById('mapHint');
    const routeCtrl = document.getElementById('routeControls');
    const goalCtrl = document.getElementById('goalControls');

    originEl.textContent = navState.selectedOrigin || '—';
    destEl.textContent = navState.selectedDest || '—';
    startBtn.disabled = !(navState.selectedOrigin && navState.selectedDest);

    if (navState.navMode === 'route') {
        routeCtrl.style.display = 'flex';
        goalCtrl.style.display = 'none';
        hint.style.display = 'block';
        if (navState.routeNodes.length === 0) {
            hint.textContent = '⏳ 節點載入中…';
        } else {
            hint.textContent = navState.selectionStep === 0
                ? '📍 點選節點選起點'
                : '📍 再點選節點選終點';
        }
    } else {
        routeCtrl.style.display = 'none';
        goalCtrl.style.display = 'flex';
        hint.style.display = 'block';
        hint.textContent = '📍 雙擊地圖任意位置設定目標';
    }

    // 更新節點 marker 的選中狀態
    document.querySelectorAll('.map-node').forEach(m => {
        m.classList.toggle('origin', m.dataset.name === navState.selectedOrigin);
        m.classList.toggle('dest', m.dataset.name === navState.selectedDest);
    });
}
