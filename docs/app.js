// 가위바위보 인식 웹 데모
// Python 버전(custom_gesture_webcam.py)과 같은 순서로 처리한다:
//   좌우 반전 프레임 -> Hand Landmarker -> 특징(gesture_features.py) -> MLP(model.json) -> 스무딩/임계값
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

const EMOJI = { rock: "🪨", scissors: "✂️", paper: "✋" };
const EMOJI_FONT = '"Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif';
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const video = document.createElement("video");
video.playsInline = true;
video.muted = true;

let model = null;
let landmarker = null;
let lastVideoTime = -1;
let lastTs = 0;
const history = new Map(); // handedness -> 최근 확률 배열들

// ---------------------------------------------------------------- 모델
function predictProba(feature) {
  let a = feature.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  model.layers.forEach((layer, li) => {
    const { W, b } = layer;
    const out = b.slice();
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
    }
    a = li < model.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}

// gesture_features.py 의 landmarks_to_row + row_to_feature 와 동일
function toFeature(landmarks, w, h, hand) {
  const x0 = landmarks[0].x * w, y0 = landmarks[0].y * h, z0 = landmarks[0].z * w;
  const pts = landmarks.map((l) => [l.x * w - x0, l.y * h - y0, l.z * w - z0]);
  if (hand === "Left") pts.forEach((p) => (p[0] = -p[0]));
  const scale = Math.max(...pts.map((p) => Math.hypot(p[0], p[1])));
  const k = scale > 1e-6 ? 1 / scale : 1;
  return pts.flatMap((p) => [p[0] * k, p[1] * k, p[2] * k]);
}

function selfCheck() {
  if (!model.checks?.length) return;
  let maxErr = 0;
  for (const c of model.checks) {
    const p = predictProba(c.x);
    p.forEach((v, i) => (maxErr = Math.max(maxErr, Math.abs(v - c.proba[i]))));
  }
  const ok = maxErr < 1e-4;
  const el = $("check");
  el.textContent = ok ? "모델 검증 통과" : `모델 검증 오차 ${maxErr.toExponential(1)}`;
  el.className = ok ? "check-ok" : "check-bad";
  console.log(`[gesture] self-check max error = ${maxErr}`);
}

function buildBars() {
  $("bars").innerHTML = model.labels
    .map((l, i) => `<div class="bar-row"><span>${EMOJI[l] ?? ""} ${l}</span>` +
      `<div class="bar"><i id="bar${i}"></i></div><span class="pct" id="pct${i}">0%</span></div>`)
    .join("");
}

// ---------------------------------------------------------------- 그리기
function drawHand(pts, color) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawLabel(text, x, y) {
  ctx.font = "600 20px 'Malgun Gothic', system-ui, sans-serif";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(20,20,20,0.85)";
  ctx.fillRect(x - 4, y - 24, w + 12, 30);
  ctx.fillStyle = "#ffe14d";
  ctx.fillText(text, x + 2, y);
}

function showResult(first) {
  if (!first) {
    $("emoji").textContent = "";
    $("name").textContent = "손 없음";
    $("sub").textContent = "손을 보여주세요";
    model.labels.forEach((_, i) => {
      $(`bar${i}`).style.width = "0%";
      $(`pct${i}`).textContent = "0%";
    });
    return;
  }
  const { name, avg, best } = first;
  $("emoji").textContent = EMOJI[name] ?? "";
  $("name").textContent = name;
  $("sub").textContent = `확신도 ${(avg[best] * 100).toFixed(0)}%`;
  avg.forEach((p, i) => {
    $(`bar${i}`).style.width = `${(p * 100).toFixed(1)}%`;
    $(`pct${i}`).textContent = `${(p * 100).toFixed(0)}%`;
  });
}

// ---------------------------------------------------------------- 메인 루프
function frame() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const w = video.videoWidth, h = video.videoHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    // Python 버전처럼 좌우 반전된 프레임으로 인식 (학습 데이터와 조건 일치)
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    const ts = Math.max(performance.now(), lastTs + 1);
    lastTs = ts;
    const result = landmarker.detectForVideo(canvas, ts);

    const threshold = Number($("thr").value);
    const smooth = Number($("smooth").value);
    const seen = new Set();
    let first = null;

    result.landmarks.forEach((landmarks, k) => {
      const hand = result.handedness[k][0].categoryName;
      seen.add(hand);
      const proba = predictProba(toFeature(landmarks, w, h, hand));

      const hist = history.get(hand) ?? [];
      hist.push(proba);
      while (hist.length > smooth) hist.shift();
      history.set(hand, hist);
      const avg = proba.map((_, i) => hist.reduce((s, p) => s + p[i], 0) / hist.length);
      const best = avg.indexOf(Math.max(...avg));
      const name = avg[best] >= threshold ? model.labels[best] : "Unknown";
      if (!first) first = { name, avg, best };

      const pts = landmarks.map((l) => [l.x * w, l.y * h]);
      drawHand(pts, name === "Unknown" ? "#a0a0a0" : "#2bd66b");
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const minX = Math.min(...xs), minY = Math.min(...ys), maxX = Math.max(...xs);
      drawLabel(`${hand}: ${name} (${avg[best].toFixed(2)})`, minX, Math.max(28, minY - 12));
      if (EMOJI[name]) {
        ctx.font = `96px ${EMOJI_FONT}`;
        ctx.textBaseline = "top";
        ctx.fillText(EMOJI[name], Math.min(maxX + 12, w - 110), Math.max(0, minY));
        ctx.textBaseline = "alphabetic";
      }
    });
    for (const hand of [...history.keys()]) if (!seen.has(hand)) history.delete(hand);
    showResult(first);
  }
  requestAnimationFrame(frame);
}

async function startCamera() {
  const btn = $("start");
  btn.disabled = true;
  btn.textContent = "카메라 연결 중…";
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    await video.play();
    $("overlay").remove();
    $("status").textContent = "실행 중 · rock / scissors / paper 를 보여주세요";
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = "다시 시도";
    $("status").textContent =
      err.name === "NotAllowedError" ? "카메라 권한이 거부되었습니다. 주소창의 카메라 아이콘에서 허용해 주세요."
      : err.name === "NotFoundError" ? "카메라를 찾을 수 없습니다."
      : `카메라를 열 수 없습니다: ${err.message}`;
  }
}

// ---------------------------------------------------------------- 초기화
$("thr").addEventListener("input", (e) => ($("thrOut").textContent = Number(e.target.value).toFixed(2)));
$("smooth").addEventListener("input", (e) => ($("smoothOut").textContent = e.target.value));
$("start").addEventListener("click", startCamera);

async function init() {
  const status = $("status");
  try {
    status.textContent = "모델 불러오는 중…";
    const [m, vision] = await Promise.all([
      fetch("model.json").then((r) => {
        if (!r.ok) throw new Error(`model.json ${r.status}`);
        return r.json();
      }),
      FilesetResolver.forVisionTasks(WASM_PATH),
    ]);
    model = m;
    buildBars();
    selfCheck();
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: HAND_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
    });
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
    } catch (e) {
      console.warn("GPU delegate 실패, CPU로 재시도", e);
      landmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
    }
    status.textContent = `준비 완료 · 클래스: ${model.labels.join(", ")}`;
    $("start").disabled = false;
    $("start").textContent = "📷 카메라 시작";
  } catch (err) {
    console.error(err);
    status.textContent = `불러오기 실패: ${err.message}`;
    $("start").textContent = "불러오기 실패";
  }
}
init();
