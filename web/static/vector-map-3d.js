// 3D vector map — Three.js (this project's one and only frontend
// dependency, vendored locally in vendor/three/, no CDN at runtime, no
// build step). Renders the same PCA-projected chunk positions the SVG
// version used to (see src/retriever.py's _fit_pca — now 3 components
// instead of 2), with real orbit controls and an on-screen distance label
// that actually works on hover *and* touch, unlike the old SVG <title>
// tooltip it replaces.
//
// Bridges into the classic (non-module) app.js via a small explicit
// global assigned once at the bottom of this module — app.js never
// imports Three.js itself, it only calls window.VectorMap3D.render(...).
import * as THREE from "three";
import { OrbitControls } from "/vendor/three/OrbitControls.js";

const COLOR_BG = 0x0f1117;      // --bg
const COLOR_MUTED = 0x8a90a3;   // --text-muted
const COLOR_ACCENT = 0x6ea8fe;  // --accent
const COLOR_WARN = 0xfbbf24;    // --warn

// Shared across every render, sized as unit (radius-1) spheres — actual
// on-screen size is set per-mesh via mesh.scale, proportional to the
// current store's PCA spread (see radiusFor() below). Fixed absolute
// radii don't work here: PCA-projected coordinates for a small KB can sit
// within a few hundredths of a unit of each other (confirmed by testing —
// the real reference-to-top-K distances for this demo's 21-chunk index
// are often smaller than a fixed 0.1-0.2 radius), which made same-sized
// spheres swallow each other whole and hide the connecting lines inside
// them entirely.
const geomOther = new THREE.SphereGeometry(1, 12, 12);
const geomTopK = new THREE.SphereGeometry(1, 16, 16);
const geomReference = new THREE.SphereGeometry(1, 16, 16);
const matOther = new THREE.MeshBasicMaterial({ color: COLOR_MUTED });
const matTopK = new THREE.MeshBasicMaterial({ color: COLOR_ACCENT });
const matReference = new THREE.MeshBasicMaterial({ color: COLOR_WARN });

const RADIUS_FACTOR_OTHER = 0.015;
const RADIUS_FACTOR_TOPK = 0.028;
const RADIUS_FACTOR_REFERENCE = 0.035;

// Visible spheres are deliberately small (see above), which would make
// them hard to hit precisely — especially on touch, with no cursor
// precision at all. Each point gets a second, larger, fully invisible
// sphere at the same position, used only for raycasting; the small
// visible sphere is purely decorative and never itself raycast against.
const HIT_RADIUS_FACTOR = 0.05;
const geomHit = new THREE.SphereGeometry(1, 8, 8);
const matHit = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });

let scene, camera, renderer, controls, raycaster, pointer, resizeObserver;
let initialized = false;
let currentContainer = null;
let labelEl = null;
let pickableMeshes = []; // invisible larger hit-test spheres, one per chunk point — what the raycaster actually tests against
let visibleMeshes = [];  // the small decorative spheres actually drawn on screen, kept separately so clearScene() can remove both sets
let onPointClick = null; // (source, chunkIndex) => void, supplied fresh by each render() call
let currentReferenceLabel = "";
let lastAllScores = null;
let lastReferencePoint = null;

function key(source, chunkIndex) {
  return `${source}#${chunkIndex}`;
}

function renderFrame() {
  if (renderer) renderer.render(scene, camera);
}

function ensureInit(container) {
  if (initialized) return true;
  currentContainer = container;

  let webglOk = true;
  try {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(COLOR_BG);
    scene.fog = new THREE.Fog(COLOR_BG, 8, 40); // depth cue: farther points fade toward the background

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 10);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
  } catch (err) {
    webglOk = false;
  }
  if (!webglOk || !renderer) {
    container.innerHTML = '<p class="muted">3D view unavailable in this browser (no WebGL).</p>';
    return false;
  }

  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false; // render-on-change instead of a rAF loop — this is a static ~20-100 point scene, no need for a continuous loop
  controls.addEventListener("change", renderFrame);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2();

  labelEl = document.createElement("div");
  labelEl.className = "vector-map-label";
  labelEl.hidden = true;
  container.appendChild(labelEl);

  resizeObserver = new ResizeObserver(() => onResize());
  resizeObserver.observe(container);

  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("click", onClick);
  renderer.domElement.addEventListener("touchstart", onTouchTap, { passive: true });

  initialized = true;
  return true;
}

function onResize() {
  if (!currentContainer || !renderer) return;
  const w = currentContainer.clientWidth, h = currentContainer.clientHeight;
  if (w === 0 || h === 0) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderFrame();
}

function clearScene() {
  pickableMeshes.forEach((mesh) => scene.remove(mesh));
  pickableMeshes = [];
  visibleMeshes.forEach((mesh) => scene.remove(mesh));
  visibleMeshes = [];
  const oldRef = scene.getObjectByName("vector-map-reference");
  if (oldRef) scene.remove(oldRef);
  const oldLines = scene.getObjectByName("vector-map-lines");
  if (oldLines) {
    scene.remove(oldLines);
    oldLines.geometry.dispose();
    oldLines.material.dispose();
  }
}

function nearestSet(allScores, topKSources) {
  const set = new Set();
  const remaining = new Map();
  topKSources.forEach((src) => remaining.set(src, (remaining.get(src) || 0) + 1));
  allScores.forEach((s) => {
    const left = remaining.get(s.source) || 0;
    if (left > 0) {
      set.add(key(s.source, s.chunk_index));
      remaining.set(s.source, left - 1);
    }
  });
  return set;
}

// Shared by populateScene (sphere sizing) and frameCamera (camera
// distance) so both stay proportional to the same notion of "how big is
// this data" — computed fresh per render since it depends on the actual
// PCA spread of whichever store/session is active.
function computeBounds(allScores, referencePoint) {
  const xs = allScores.map((s) => s.x).concat([referencePoint.x]);
  const ys = allScores.map((s) => s.y).concat([referencePoint.y]);
  const zs = allScores.map((s) => s.z).concat([referencePoint.z]);
  const center = new THREE.Vector3(
    (Math.min(...xs) + Math.max(...xs)) / 2,
    (Math.min(...ys) + Math.max(...ys)) / 2,
    (Math.min(...zs) + Math.max(...zs)) / 2
  );
  const spread = Math.max(
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
    Math.max(...zs) - Math.min(...zs),
    0.05 // floor so a near-degenerate point cloud doesn't collapse sphere sizes to ~0
  );
  return { center, spread };
}

function populateScene(allScores, topKSources, referencePoint, spread) {
  const topKSet = nearestSet(allScores, topKSources);

  allScores.forEach((s) => {
    const isTopK = topKSet.has(key(s.source, s.chunk_index));
    const userData = { source: s.source, chunk_index: s.chunk_index, score: s.score };

    const mesh = new THREE.Mesh(isTopK ? geomTopK : geomOther, isTopK ? matTopK : matOther);
    mesh.position.set(s.x, s.y, s.z);
    mesh.scale.setScalar(spread * (isTopK ? RADIUS_FACTOR_TOPK : RADIUS_FACTOR_OTHER));
    scene.add(mesh);
    visibleMeshes.push(mesh);

    const hitMesh = new THREE.Mesh(geomHit, matHit);
    hitMesh.position.copy(mesh.position);
    hitMesh.scale.setScalar(spread * HIT_RADIUS_FACTOR);
    hitMesh.userData = userData;
    scene.add(hitMesh);
    pickableMeshes.push(hitMesh);
  });

  const reference = new THREE.Mesh(geomReference, matReference);
  reference.name = "vector-map-reference";
  reference.position.set(referencePoint.x, referencePoint.y, referencePoint.z);
  reference.scale.setScalar(spread * RADIUS_FACTOR_REFERENCE);
  scene.add(reference);

  const linePositions = [];
  allScores.forEach((s) => {
    if (!topKSet.has(key(s.source, s.chunk_index))) return;
    linePositions.push(referencePoint.x, referencePoint.y, referencePoint.z, s.x, s.y, s.z);
  });
  if (linePositions.length > 0) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    // Solid, not dashed: at this scene's scale the reference-to-top-K
    // segments are often shorter than a single dash+gap cycle (points can
    // sit within a couple hundredths of a unit of each other), so
    // LineDashedMaterial rendered nothing visible in practice — confirmed
    // by screenshot during testing. A plain solid line is the reliable
    // choice here; WebGL also caps line width at ~1px on most platforms
    // regardless, so pixel-perfect parity with the old SVG dashes was
    // never on the table.
    const mat = new THREE.LineBasicMaterial({ color: COLOR_ACCENT, transparent: true, opacity: 0.6 });
    const lines = new THREE.LineSegments(geom, mat);
    lines.name = "vector-map-lines";
    scene.add(lines);
  }
}

function frameCamera(center, spread) {
  const distance = spread * 1.9 + 2;
  camera.position.set(center.x, center.y, center.z + distance);
  camera.up.set(0, 1, 0);
  controls.target.copy(center);
  controls.update();
}

function pickAt(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickableMeshes, false);
  return hits.length > 0 ? hits[0].object : null;
}

function showLabel(hit, clientX, clientY, withRerootButton) {
  const rect = currentContainer.getBoundingClientRect();
  labelEl.innerHTML = "";
  const text = document.createElement("div");
  text.textContent = `${hit.userData.source} #${hit.userData.chunk_index} — similarity to ${currentReferenceLabel}: ${hit.userData.score.toFixed(3)}`;
  labelEl.appendChild(text);
  if (withRerootButton) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-chip";
    btn.textContent = "Make this the reference";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (onPointClick) onPointClick(hit.userData.source, hit.userData.chunk_index);
      labelEl.hidden = true;
    });
    labelEl.appendChild(btn);
  }
  let left = clientX - rect.left + 12;
  let top = clientY - rect.top + 12;
  labelEl.style.left = `${left}px`;
  labelEl.style.top = `${top}px`;
  labelEl.hidden = false;
}

function hideLabel() {
  if (labelEl) labelEl.hidden = true;
}

// Desktop: hover shows the label, click reroots immediately — matches the
// old SVG behavior exactly.
function onPointerMove(e) {
  const hit = pickAt(e.clientX, e.clientY);
  if (hit) showLabel(hit, e.clientX, e.clientY, false);
  else hideLabel();
}

function onClick(e) {
  const hit = pickAt(e.clientX, e.clientY);
  if (hit && onPointClick) onPointClick(hit.userData.source, hit.userData.chunk_index);
}

// Touch has no hover, and a bare tap can't safely both reveal the label
// *and* reroot (reroot destroys the scene the label just described) — so
// tapping a dot shows the label with an explicit button, and reroot only
// fires from that button.
function onTouchTap(e) {
  if (e.touches.length !== 1) return;
  const touch = e.touches[0];
  const hit = pickAt(touch.clientX, touch.clientY);
  if (hit) showLabel(hit, touch.clientX, touch.clientY, true);
  else hideLabel();
}

function render(container, allScores, topKSources, referencePoint, referenceLabel, onPointClickCallback) {
  if (!allScores || allScores.length === 0) {
    container.innerHTML = '<p class="muted">No chunks to show yet.</p>';
    return;
  }

  if (!ensureInit(container)) return; // WebGL unavailable — message already shown

  onPointClick = onPointClickCallback;
  currentReferenceLabel = referenceLabel;
  lastAllScores = allScores;
  lastReferencePoint = referencePoint;
  hideLabel();

  const { center, spread } = computeBounds(allScores, referencePoint);
  clearScene();
  populateScene(allScores, topKSources, referencePoint, spread);
  frameCamera(center, spread);
  renderFrame();
}

function resetView() {
  if (!initialized || !lastAllScores || !lastReferencePoint) return;
  const { center, spread } = computeBounds(lastAllScores, lastReferencePoint);
  frameCamera(center, spread);
  renderFrame();
}

window.VectorMap3D = { render, resetView };
