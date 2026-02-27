import * as THREE from "./lib/three.module.js";
import { OrbitControls } from "./lib/OrbitControls.js";

const canvas = document.getElementById("app");
const stepEl = document.getElementById("step");
const actionEl = document.getElementById("action");
const avgEnergyEl = document.getElementById("avgEnergy");
const stageEl = document.getElementById("stage");
const modeEl = document.getElementById("mode");
const riskThresholdEl = document.getElementById("riskThreshold");
const riskThresholdValueEl = document.getElementById("riskThresholdValue");
const helpBtn = document.getElementById("helpBtn");
const helpModal = document.getElementById("helpModal");
const helpCard = document.getElementById("helpCard");
const closeHelpBtn = document.getElementById("closeHelpBtn");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog("#220b11", 35, 165);

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(24, 18, 34);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 4, 0);
controls.enablePan = true;
controls.enableZoom = true;
controls.enableRotate = true;
controls.minDistance = 0.25;
controls.maxDistance = 220;
controls.zoomSpeed = 1.35;
controls.panSpeed = 1.15;
controls.rotateSpeed = 0.9;
controls.screenSpacePanning = true;

scene.add(new THREE.AmbientLight("#ffd4dd", 0.6));
const keyLight = new THREE.DirectionalLight("#fff0f4", 1.05);
keyLight.position.set(22, 34, 14);
scene.add(keyLight);
const fill = new THREE.PointLight("#ea8da7", 0.7, 220);
fill.position.set(-22, 6, 8);
scene.add(fill);
const lumenGlow = new THREE.PointLight("#ffaec2", 0.5, 130);
lumenGlow.position.set(0, 8, 0);
scene.add(lumenGlow);

const anatomyGroup = new THREE.Group();
scene.add(anatomyGroup);
const lesionGroup = new THREE.Group();
scene.add(lesionGroup);
const lymphGroup = new THREE.Group();
scene.add(lymphGroup);
const metastasisGroup = new THREE.Group();
scene.add(metastasisGroup);
const moistureGroup = new THREE.Group();
scene.add(moistureGroup);

const nodes = [];
const nodeGroup = new THREE.Group();
scene.add(nodeGroup);

const baseEdgeStyle = { transparent: true, opacity: 0.22 };
const edgeGroup = new THREE.Group();
scene.add(edgeGroup);

let simulationStep = 0;
let running = true;
let useDatasetPlayback = false;
let dataset = null;
let datasetSource = "synthetic";
let datasetStepIndex = 0;
let lastStepAtMs = 0;
let progressedSteps = 0;
let modeLabel = "Mode: synthetic fallback";
let spreadThreshold = 0.45;
let globalStage = 0;

let activeCameraPreset = "overview";
const cameraTargetPos = new THREE.Vector3(24, 18, 34);
const cameraTargetLook = new THREE.Vector3(0, 4, 0);
let flyPhase = 0;
let lumenOrbitAngle = 0;
let keyboardWalkEnabled = true;
const pressedKeys = new Set();
const walkVelocity = new THREE.Vector3();
let cameraAutoBlend = false;
let userIsDragging = false;

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let cycleRecording = false;
let cycleTargetSteps = 0;
let helpOpen = false;

const defaultCellCount = 240;
const tubeRadius = 7.4;
const lesionMarkers = [];
const lymphNodes = [];
const metastasisNodes = [];
const moistureStreaks = [];

const colonCurve = new THREE.CatmullRomCurve3([
  new THREE.Vector3(-16, 2, -2),
  new THREE.Vector3(-12, 14, -4),
  new THREE.Vector3(-4, 18, 6),
  new THREE.Vector3(6, 14, 8),
  new THREE.Vector3(12, 8, 2),
  new THREE.Vector3(10, -2, -6),
  new THREE.Vector3(2, -10, -8),
  new THREE.Vector3(-8, -14, -2),
  new THREE.Vector3(-14, -18, 8),
]);

const textureLoader = new THREE.TextureLoader();
const colonMucosaTexture = textureLoader.load("./colon.jpg");
colonMucosaTexture.wrapS = THREE.RepeatWrapping;
colonMucosaTexture.wrapT = THREE.RepeatWrapping;
colonMucosaTexture.repeat.set(3, 9);
colonMucosaTexture.colorSpace = THREE.SRGBColorSpace;

function baseFrames(t) {
  const point = colonCurve.getPointAt(t);
  const tangent = colonCurve.getTangentAt(t).normalize();
  const up = Math.abs(tangent.dot(new THREE.Vector3(0, 1, 0))) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const normal = new THREE.Vector3().crossVectors(tangent, up).normalize();
  const binormal = new THREE.Vector3().crossVectors(tangent, normal).normalize();
  return { point, tangent, normal, binormal };
}

function buildColorectalAnatomy() {
  anatomyGroup.clear();
  moistureGroup.clear();
  moistureStreaks.length = 0;

  const outerTube = new THREE.Mesh(
    new THREE.TubeGeometry(colonCurve, 220, tubeRadius, 34, false),
    new THREE.MeshPhysicalMaterial({
      color: "#d9869d", roughness: 0.5, transmission: 0.28, thickness: 1.2,
      transparent: true, opacity: 0.52, side: THREE.DoubleSide,
    })
  );
  anatomyGroup.add(outerTube);

  const mucosaTube = new THREE.Mesh(
    new THREE.TubeGeometry(colonCurve, 220, tubeRadius - 1.0, 30, false),
    new THREE.MeshPhysicalMaterial({
      map: colonMucosaTexture,
      bumpMap: colonMucosaTexture,
      bumpScale: 0.46,
      color: "#ffd1da",
      roughness: 0.32,
      metalness: 0.02,
      transmission: 0.08,
      clearcoat: 0.96,
      clearcoatRoughness: 0.12,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.92,
      emissive: "#3a101a",
      emissiveIntensity: 0.24,
    })
  );
  anatomyGroup.add(mucosaTube);

  // Thin wet sheen layer to mimic endoscopic specular glare on mucosa.
  const sheenTube = new THREE.Mesh(
    new THREE.TubeGeometry(colonCurve, 220, tubeRadius - 0.92, 28, false),
    new THREE.MeshPhysicalMaterial({
      color: "#ffd8df",
      roughness: 0.12,
      metalness: 0.0,
      transmission: 0.35,
      thickness: 0.15,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.2,
    })
  );
  anatomyGroup.add(sheenTube);

  const depthCore = new THREE.Mesh(
    new THREE.TubeGeometry(colonCurve, 220, tubeRadius - 3.0, 22, false),
    new THREE.MeshBasicMaterial({
      color: "#2a0a10",
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.62,
    })
  );
  anatomyGroup.add(depthCore);

  for (let i = 0; i < 44; i += 1) {
    const t = (i + 0.5) / 44;
    const f = baseFrames(t);
    const angle = (i * 0.73) % (Math.PI * 2);
    const radial = f.normal.clone().multiplyScalar(Math.cos(angle)).add(f.binormal.clone().multiplyScalar(Math.sin(angle))).normalize();
    const p = f.point.clone().add(radial.multiplyScalar(tubeRadius - 0.72));
    const drop = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 10, 10),
      new THREE.MeshPhysicalMaterial({
        color: "#ffdbe3",
        roughness: 0.04,
        transmission: 0.65,
        thickness: 0.12,
        clearcoat: 1.0,
        clearcoatRoughness: 0.02,
        transparent: true,
        opacity: 0.5,
      })
    );
    drop.position.copy(p);
    drop.scale.set(1.9, 0.55, 0.55);
    moistureGroup.add(drop);
    moistureStreaks.push({ mesh: drop, phase: Math.random() * Math.PI * 2 });
  }

  const foldMaterial = new THREE.MeshStandardMaterial({ color: "#cb6d87", roughness: 0.9, transparent: true, opacity: 0.58 });
  for (let i = 0; i < 34; i += 1) {
    const t = (i + 1) / 35;
    const { point, tangent, normal, binormal } = baseFrames(t);
    const fold = new THREE.Mesh(new THREE.TorusGeometry(tubeRadius - 0.6, 0.34, 10, 44), foldMaterial);
    const matrix = new THREE.Matrix4();
    matrix.makeBasis(normal, binormal, tangent);
    fold.quaternion.setFromRotationMatrix(matrix);
    fold.position.copy(point);
    fold.scale.set(1, 0.7 + 0.5 * Math.sin(i * 1.3), 1);
    anatomyGroup.add(fold);
  }

  // Build rectum as a continuation of colon endpoint to avoid detached geometry.
  const endF = baseFrames(1.0);
  const p0 = endF.point.clone();
  const p1 = p0.clone().add(endF.tangent.clone().multiplyScalar(6.5));
  const p2 = p1.clone().add(endF.tangent.clone().multiplyScalar(5.5)).add(endF.normal.clone().multiplyScalar(1.1));
  const p3 = p2.clone().add(endF.tangent.clone().multiplyScalar(4.2)).add(new THREE.Vector3(0, -1.8, 0));
  const rectumCurve = new THREE.CatmullRomCurve3([p0, p1, p2, p3]);

  const rectumOuter = new THREE.Mesh(
    new THREE.TubeGeometry(rectumCurve, 84, tubeRadius - 0.35, 28, false),
    new THREE.MeshPhysicalMaterial({
      color: "#c55f79",
      roughness: 0.5,
      transmission: 0.22,
      transparent: true,
      opacity: 0.48,
      side: THREE.DoubleSide,
    })
  );
  anatomyGroup.add(rectumOuter);

  const rectumMucosa = new THREE.Mesh(
    new THREE.TubeGeometry(rectumCurve, 84, tubeRadius - 1.15, 24, false),
    new THREE.MeshPhysicalMaterial({
      map: colonMucosaTexture,
      bumpMap: colonMucosaTexture,
      bumpScale: 0.42,
      color: "#ffd1da",
      roughness: 0.34,
      transmission: 0.1,
      clearcoat: 0.95,
      clearcoatRoughness: 0.1,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.9,
      emissive: "#3a101a",
      emissiveIntensity: 0.2,
    })
  );
  anatomyGroup.add(rectumMucosa);
}

function buildStageAnatomyTargets() {
  lymphGroup.clear();
  metastasisGroup.clear();
  lymphNodes.length = 0;
  metastasisNodes.length = 0;

  const lymphMat = new THREE.MeshStandardMaterial({ color: "#9adf86", emissive: "#376b2f", emissiveIntensity: 0.15, transparent: true, opacity: 0.72 });
  const metaMat = new THREE.MeshStandardMaterial({ color: "#f9c26b", emissive: "#7f4b1d", emissiveIntensity: 0.2, transparent: true, opacity: 0.75 });

  for (let i = 0; i < 14; i += 1) {
    const t = 0.08 + 0.84 * (i / 13);
    const f = baseFrames(t);
    const pos = f.point.clone().add(f.normal.clone().multiplyScalar(tubeRadius + 1.8 + 0.6 * Math.sin(i)));
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.36, 14, 14), lymphMat.clone());
    mesh.position.copy(pos);
    mesh.visible = false;
    lymphGroup.add(mesh);
    lymphNodes.push(mesh);
  }

  const metaTargets = [
    new THREE.Vector3(24, 16, -2),
    new THREE.Vector3(26, 11, 6),
    new THREE.Vector3(22, 8, -8),
    new THREE.Vector3(-25, 15, 12),
    new THREE.Vector3(-22, 10, 20),
  ];
  for (const p of metaTargets) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.62, 16, 16), metaMat.clone());
    mesh.position.copy(p);
    mesh.visible = false;
    metastasisGroup.add(mesh);
    metastasisNodes.push(mesh);
  }
}

function stageLabel(stage) {
  if (stage <= 0) return "Stage: 0 (In situ mucosa)";
  if (stage === 1) return "Stage: I (Submucosa invasion)";
  if (stage === 2) return "Stage: II (Muscle layer invasion)";
  if (stage === 3) return "Stage: III (Lymph node spread)";
  return "Stage: IV (Distant metastasis)";
}

function layerDepthForStage(stage) {
  // inward offsets from mucosal surface
  if (stage <= 0) return 0.0;    // mucosa
  if (stage === 1) return 0.14;  // submucosa
  if (stage === 2) return 0.3;   // muscularis
  if (stage === 3) return 0.44;  // beyond muscle
  return 0.6;                    // serosa/outside
}

function lesionColor(v) {
  const t = THREE.MathUtils.clamp(v, 0, 1);
  return new THREE.Color().setHSL(0.12 * (1 - t), 1, 0.55 - 0.15 * t);
}

function initLesionOverlay(count = 140) {
  lesionGroup.clear();
  lesionMarkers.length = 0;
  const geom = new THREE.SphereGeometry(0.18, 10, 10);

  for (let i = 0; i < count; i += 1) {
    const t = (i + Math.random() * 0.7) / count;
    const { point, normal, binormal } = baseFrames(t);
    const angle = Math.random() * Math.PI * 2;
    const radial = normal.clone().multiplyScalar(Math.cos(angle)).add(binormal.clone().multiplyScalar(Math.sin(angle))).normalize();
    const pos = point.clone().add(radial.multiplyScalar(tubeRadius - 0.22));
    const susceptibility = Math.random() * 0.45 + 0.55;

    const mesh = new THREE.Mesh(
      geom,
      new THREE.MeshStandardMaterial({ color: lesionColor(0.1), emissive: "#2f0f11", emissiveIntensity: 0.4, transparent: true, opacity: 0.8 })
    );
    mesh.position.copy(pos);
    lesionGroup.add(mesh);

    lesionMarkers.push({ pos, susceptibility, mesh });
  }
}

function updateLesionOverlay() {
  if (nodes.length === 0) return;
  for (const marker of lesionMarkers) {
    let acc = 0;
    let wsum = 0;
    for (const n of nodes) {
      const d = marker.pos.distanceTo(n.root.position);
      if (d > 8.2) continue;
      const w = 1 / (0.7 + d * d);
      acc += n.energy * w;
      wsum += w;
    }
    const local = wsum > 0 ? acc / wsum : 0;
    const lesion = THREE.MathUtils.clamp(local * marker.susceptibility, 0, 1);
    marker.mesh.material.color.copy(lesionColor(lesion));
    marker.mesh.material.emissive.copy(lesionColor(lesion)).multiplyScalar(0.35 + lesion * 0.55);
    marker.mesh.scale.setScalar(0.7 + lesion * 1.4);
    marker.mesh.visible = lesion >= spreadThreshold;
  }
}

const cellProfiles = {
  colonocyte: { radius: 0.48, scale: new THREE.Vector3(0.82, 1.34, 0.82), membrane: "#f6c4cf", nucleus: "#613f74", organelle: "#cf7f8f" },
  goblet: { radius: 0.6, scale: new THREE.Vector3(0.95, 1.22, 0.95), membrane: "#ffdce4", nucleus: "#60457c", organelle: "#db90a1" },
  stem: { radius: 0.52, scale: new THREE.Vector3(0.9, 1.05, 0.9), membrane: "#f0b3c4", nucleus: "#4a3568", organelle: "#c97287" },
  tumor: { radius: 0.57, scale: new THREE.Vector3(1.1, 1.05, 1.1), membrane: "#d57890", nucleus: "#3b214f", organelle: "#9f4d63" },
};

function inferCellType(i) {
  if (i % 17 === 0) return "tumor";
  if (i % 9 === 0) return "stem";
  if (i % 6 === 0) return "goblet";
  return "colonocyte";
}

function clearSceneGraph() {
  for (const n of nodes) {
    n.membrane.geometry.dispose();
    n.membrane.material.dispose();
    n.nucleus.geometry.dispose();
    n.nucleus.material.dispose();
    for (const o of n.organelles) {
      o.geometry.dispose();
      o.material.dispose();
    }
    if (Array.isArray(n.lesionLobes)) {
      for (const l of n.lesionLobes) {
        l.geometry.dispose();
        l.material.dispose();
      }
    }
    if (n.ulcerMesh) {
      n.ulcerMesh.geometry.dispose();
      n.ulcerMesh.material.dispose();
    }
  }
  nodes.length = 0;
  nodeGroup.clear();
  for (const line of edgeGroup.children) {
    line.geometry.dispose();
    line.material.dispose();
  }
  edgeGroup.clear();
}

function energyToTone(type, energy) {
  const e = THREE.MathUtils.clamp(energy, 0, 1);
  const c = new THREE.Color(cellProfiles[type].membrane);
  const hsl = { h: 0, s: 0, l: 0 };
  c.getHSL(hsl);
  return new THREE.Color().setHSL(hsl.h - 0.01 * e, Math.min(1, hsl.s + 0.15 * e), Math.max(0.26, hsl.l - 0.17 * e));
}

function colonAnchor(index, total) {
  const t = (index + 0.5) / total;
  const { point, tangent, normal, binormal } = baseFrames(t);
  const angle = ((index * 2.3999632297) % (Math.PI * 2));
  const r = tubeRadius - 0.28 + (Math.random() - 0.5) * 0.22;
  const radial = normal.clone().multiplyScalar(Math.cos(angle)).add(binormal.clone().multiplyScalar(Math.sin(angle))).normalize();
  return { anchor: point.clone().add(radial.multiplyScalar(r)), tangent, radial };
}

function buildCell(type, energy) {
  const p = cellProfiles[type];
  const root = new THREE.Group();

  const membrane = new THREE.Mesh(
    new THREE.SphereGeometry(p.radius, 24, 24),
    new THREE.MeshPhysicalMaterial({
      color: energyToTone(type, energy), roughness: 0.35, transmission: 0.3, thickness: 0.48,
      clearcoat: 0.2, clearcoatRoughness: 0.45, transparent: true, opacity: 0.93,
    })
  );
  membrane.scale.copy(p.scale);

  const nucleus = new THREE.Mesh(
    new THREE.SphereGeometry(p.radius * 0.36, 18, 18),
    new THREE.MeshStandardMaterial({ color: p.nucleus, roughness: 0.58, emissive: "#261330", emissiveIntensity: 0.2 })
  );
  nucleus.position.set((Math.random() - 0.5) * 0.08, (Math.random() - 0.5) * 0.08, 0);
  membrane.add(nucleus);

  const organelles = [];
  const organelleCount = type === "tumor" ? 10 : 6;
  for (let i = 0; i < organelleCount; i += 1) {
    const mito = new THREE.Mesh(new THREE.SphereGeometry(p.radius * 0.12, 10, 10), new THREE.MeshStandardMaterial({ color: p.organelle, roughness: 0.4 }));
    mito.scale.set(1.7, 0.8, 0.8);
    const rr = p.radius * 0.3;
    mito.position.set((Math.random() - 0.5) * rr, (Math.random() - 0.5) * rr, (Math.random() - 0.5) * rr);
    mito.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
    membrane.add(mito);
    organelles.push(mito);
  }

  let lesionLobes = [];
  let ulcerMesh = null;
  if (type === "tumor") {
    // Replace spherical tumor look with endoscopic-like lobulated lesion morphology.
    membrane.scale.set(1.28, 1.08, 1.28);
    membrane.material.color = new THREE.Color("#d37b6c");
    membrane.material.roughness = 0.28;
    membrane.material.clearcoat = 1.0;
    membrane.material.clearcoatRoughness = 0.06;
    membrane.material.transmission = 0.2;

    const lobeMat = new THREE.MeshPhysicalMaterial({
      color: "#d18473",
      roughness: 0.23,
      transmission: 0.18,
      thickness: 0.22,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04,
      transparent: true,
      opacity: 0.95,
    });
    const lobeOffsets = [
      [0.36, 0.04, 0.0],
      [-0.28, 0.09, 0.18],
      [0.05, 0.18, -0.26],
      [-0.08, -0.02, -0.1],
    ];
    for (let i = 0; i < lobeOffsets.length; i += 1) {
      const lobe = new THREE.Mesh(new THREE.SphereGeometry(p.radius * 0.62, 20, 20), lobeMat.clone());
      const o = lobeOffsets[i];
      lobe.position.set(o[0], o[1], o[2]);
      lobe.scale.set(1.0 + 0.22 * (i % 2), 0.8 + 0.18 * ((i + 1) % 2), 1.0);
      membrane.add(lobe);
      lesionLobes.push(lobe);
    }

    ulcerMesh = new THREE.Mesh(
      new THREE.SphereGeometry(p.radius * 0.24, 14, 14),
      new THREE.MeshStandardMaterial({
        color: "#7f2a2a",
        roughness: 0.74,
        metalness: 0.02,
        emissive: "#2a0a0a",
        emissiveIntensity: 0.12,
      })
    );
    ulcerMesh.position.set(0.08, 0.03, 0.26);
    ulcerMesh.scale.set(1.4, 0.7, 0.4);
    membrane.add(ulcerMesh);
  }

  root.add(membrane);
  return { root, membrane, nucleus, organelles, lesionLobes, ulcerMesh };
}

function initCellsFromSpec(specs) {
  const data = specs || [];
  const count = data.length > 0 ? data.length : defaultCellCount;

  for (let i = 0; i < count; i += 1) {
    const spec = data[i] || {};
    const energy = typeof spec.energy === "number" ? spec.energy : Math.random() * 0.9 + 0.1;
    const type = typeof spec.cellType === "string" && cellProfiles[spec.cellType] ? spec.cellType : inferCellType(i);
    const cell = buildCell(type, energy);

    const { anchor, tangent, radial } = colonAnchor(i, count);
    if (Array.isArray(spec.position) && spec.position.length === 3 && spec.position.every((v) => typeof v === "number")) {
      cell.root.position.set(spec.position[0], spec.position[1], spec.position[2]);
    } else {
      cell.root.position.copy(anchor);
    }

    const inward = radial.clone().multiplyScalar(-1).normalize();
    const up = tangent.clone();
    const forward = new THREE.Vector3().crossVectors(up, inward).normalize();
    const m = new THREE.Matrix4().makeBasis(forward, up, inward);
    cell.root.quaternion.setFromRotationMatrix(m);

    const node = {
      id: typeof spec.id === "number" ? spec.id : i,
      index: i,
      type,
      energy,
      root: cell.root,
      membrane: cell.membrane,
      nucleus: cell.nucleus,
      organelles: cell.organelles,
      lesionLobes: cell.lesionLobes,
      ulcerMesh: cell.ulcerMesh,
      surfacePosition: cell.root.position.clone(),
      inwardDir: inward.clone(),
      basePosition: cell.root.position.clone(),
      axial: (i + 0.5) / count,
      ring: i % 12,
      cryptBand: Math.floor(((i + 0.5) / count) * 10),
      velocityPhase: Math.random() * Math.PI * 2,
      stage: 0,
      neighbors: new Set(),
    };

    nodes.push(node);
    nodeGroup.add(cell.root);
  }
}

function edgeStyleByType(edgeType) {
  if (edgeType === "adherens") return { color: "#f4b3c2", opacity: 0.28 };
  if (edgeType === "crypt") return { color: "#f2d3dc", opacity: 0.24 };
  if (edgeType === "tumor") return { color: "#ff5b7f", opacity: 0.42 };
  return { color: "#c9879a", opacity: 0.2 };
}

function addEdgeByIndices(a, b, edgeType = "adherens") {
  if (a === b) return;
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const key = `${lo}-${hi}`;
  if (nodes[lo].neighbors.has(key)) return;

  nodes[lo].neighbors.add(key);
  nodes[hi].neighbors.add(key);

  const geom = new THREE.BufferGeometry().setFromPoints([nodes[lo].root.position, nodes[hi].root.position]);
  const style = edgeStyleByType(edgeType);
  const line = new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ ...baseEdgeStyle, color: style.color, opacity: style.opacity })
  );
  line.userData = { a: lo, b: hi, edgeType };
  edgeGroup.add(line);
}

function epithelialDistance(a, b) {
  const da = Math.abs(a.axial - b.axial);
  const axialDist = Math.min(da, 1 - da);
  const ringDist = Math.min(Math.abs(a.ring - b.ring), 12 - Math.abs(a.ring - b.ring));
  return { axialDist, ringDist, spatialDist: a.basePosition.distanceTo(b.basePosition) };
}

function connectionRule(a, b) {
  const { axialDist, ringDist, spatialDist } = epithelialDistance(a, b);
  const sameCrypt = a.cryptBand === b.cryptBand;
  const tumorPair = a.type === "tumor" || b.type === "tumor";
  const stemPair = a.type === "stem" || b.type === "stem";

  if (tumorPair) {
    if (spatialDist < 4.8 && axialDist < 0.12) return "tumor";
    return null;
  }
  if (stemPair && sameCrypt && spatialDist < 3.8 && axialDist < 0.08) return "crypt";
  if (sameCrypt && ringDist <= 2 && spatialDist < 3.4 && axialDist < 0.06) return "adherens";
  if (ringDist <= 1 && spatialDist < 2.8 && axialDist < 0.045) return "adherens";
  return null;
}

function initEdgesFromSpec(edges) {
  if (Array.isArray(edges) && edges.length > 0) {
    for (const e of edges) {
      if (!Array.isArray(e) || e.length !== 2) continue;
      const a = e[0];
      const b = e[1];
      if (typeof a === "number" && typeof b === "number" && a >= 0 && b >= 0 && a < nodes.length && b < nodes.length) {
        const edgeType = connectionRule(nodes[a], nodes[b]) || "adherens";
        addEdgeByIndices(a, b, edgeType);
      }
    }
    return;
  }

  for (let i = 0; i < nodes.length; i += 1) {
    let attached = 0;
    for (let j = 0; j < nodes.length; j += 1) {
      if (i === j) continue;
      const edgeType = connectionRule(nodes[i], nodes[j]);
      if (!edgeType) continue;
      addEdgeByIndices(i, j, edgeType);
      attached += 1;
      if (attached >= 8) break;
    }
  }
}

function avgEnergy() {
  return nodes.reduce((acc, n) => acc + n.energy, 0) / nodes.length;
}

function applyAction(action) {
  const coupling = 0.09;
  for (let idx = 0; idx < nodes.length; idx += 1) {
    const n = nodes[idx];
    let localMean = 0;
    let c = 0;
    for (const key of n.neighbors) {
      const [ia, ib] = key.split("-").map(Number);
      const other = ia === idx ? nodes[ib] : nodes[ia];
      localMean += other.energy;
      c += 1;
    }
    if (c > 0) localMean /= c;

    const noise = (Math.random() - 0.5) * 0.02;
    const neighborInfluence = coupling * (localMean - n.energy);
    if (action === 0) n.energy += 0.02 + neighborInfluence + noise;
    if (action === 1) n.energy += -0.02 + neighborInfluence + noise;
    if (action === 2) n.energy += -0.045 + 0.6 * neighborInfluence + noise;
    if (n.type === "tumor") n.energy += 0.006;

    n.energy = THREE.MathUtils.clamp(n.energy, 0, 1.2);
  }
}

function applyDatasetStep(stepData) {
  if (!stepData || !Array.isArray(stepData.energies)) return;
  for (let i = 0; i < nodes.length && i < stepData.energies.length; i += 1) {
    if (typeof stepData.energies[i] === "number") nodes[i].energy = THREE.MathUtils.clamp(stepData.energies[i], 0, 1.2);
  }
  simulationStep = typeof stepData.step === "number" ? stepData.step : simulationStep + 1;
  stepEl.textContent = `Step: ${simulationStep}`;
  actionEl.textContent = typeof stepData.action === "number" ? `Action: ${stepData.action}` : "Action: -";
  avgEnergyEl.textContent = `Avg Energy: ${avgEnergy().toFixed(3)}`;
}

function setCameraPreset(name, instant = false) {
  activeCameraPreset = name;

  if (name === "overview") {
    cameraTargetPos.set(24, 18, 34);
    cameraTargetLook.set(0, 4, 0);
  } else if (name === "rectum") {
    cameraTargetPos.set(-19, -19, 17);
    cameraTargetLook.set(-14, -20, 9);
  } else if (name === "lumen") {
    flyPhase = 0;
    lumenOrbitAngle = 0;
  }

  if (instant && name !== "lumen") {
    camera.position.copy(cameraTargetPos);
    controls.target.copy(cameraTargetLook);
    cameraAutoBlend = false;
    return;
  }
  cameraAutoBlend = true;
}

function updateCameraMotion(dtSec) {
  const walking = pressedKeys.size > 0;
  const inLumen = activeCameraPreset === "lumen";

  if (inLumen) {
    let drive = 0;
    if (pressedKeys.has("arrowup") || pressedKeys.has("w")) drive += 1;
    if (pressedKeys.has("arrowdown") || pressedKeys.has("s")) drive -= 1;
    if (pressedKeys.has("arrowleft") || pressedKeys.has("a")) lumenOrbitAngle -= dtSec * 1.4;
    if (pressedKeys.has("arrowright") || pressedKeys.has("d")) lumenOrbitAngle += dtSec * 1.4;

    // Move through the inside like an endoscope along the colon centerline.
    flyPhase = (flyPhase + drive * dtSec * 0.1 + 1) % 1;
    const t = flyPhase;
    const aheadT = (t + 0.008) % 1;
    const f = baseFrames(t);
    const f2 = baseFrames(aheadT);
    const radial = f.normal.clone().multiplyScalar(Math.cos(lumenOrbitAngle)).add(f.binormal.clone().multiplyScalar(Math.sin(lumenOrbitAngle))).normalize();
    const radial2 = f2.normal.clone().multiplyScalar(Math.cos(lumenOrbitAngle)).add(f2.binormal.clone().multiplyScalar(Math.sin(lumenOrbitAngle))).normalize();
    cameraTargetPos.copy(f.point).add(radial.multiplyScalar(0.22));
    cameraTargetLook.copy(f2.point).add(radial2.multiplyScalar(0.18));
    cameraAutoBlend = true;
  }

  if (userIsDragging) {
    cameraAutoBlend = false;
    return;
  }

  if (!walking || inLumen) {
    if (cameraAutoBlend) {
      const posAlpha = 1 - Math.exp(-6.5 * dtSec);
      const lookAlpha = 1 - Math.exp(-7.5 * dtSec);
      camera.position.lerp(cameraTargetPos, posAlpha);
      controls.target.lerp(cameraTargetLook, lookAlpha);
      const posDone = camera.position.distanceTo(cameraTargetPos) < 0.03;
      const lookDone = controls.target.distanceTo(cameraTargetLook) < 0.03;
      if (posDone && lookDone) cameraAutoBlend = false;
    }
    return;
  }

  // Arrow-key walk mode: move camera and target together along camera-relative axes.
  const moveSpeed = 7.5 * dtSec;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = camera.up.clone().normalize();

  walkVelocity.set(0, 0, 0);
  if (pressedKeys.has("arrowup")) walkVelocity.add(forward);
  if (pressedKeys.has("arrowdown")) walkVelocity.add(forward.clone().multiplyScalar(-1));
  if (pressedKeys.has("arrowright")) walkVelocity.add(right);
  if (pressedKeys.has("arrowleft")) walkVelocity.add(right.clone().multiplyScalar(-1));
  if (pressedKeys.has("pageup")) walkVelocity.add(up);
  if (pressedKeys.has("pagedown")) walkVelocity.add(up.clone().multiplyScalar(-1));

  if (walkVelocity.lengthSq() > 0) {
    walkVelocity.normalize().multiplyScalar(moveSpeed);
    camera.position.add(walkVelocity);
    controls.target.add(walkVelocity);
    cameraAutoBlend = false;
  }
}

function updateVisuals(time) {
  let tumorBurden = 0;
  let tumorCount = 0;
  let maxEnergy = 0;
  for (const n of nodes) {
    maxEnergy = Math.max(maxEnergy, n.energy);
    if (n.type === "tumor") {
      tumorBurden += n.energy;
      tumorCount += 1;
    }
  }
  const burden = tumorCount > 0 ? tumorBurden / (1.2 * tumorCount) : avgEnergy() / 1.2;
  if (maxEnergy > 1.08 || burden > 0.82) globalStage = 4;
  else if (burden > 0.62) globalStage = 3;
  else if (burden > 0.42) globalStage = 2;
  else if (burden > 0.22) globalStage = 1;
  else globalStage = 0;
  stageEl.textContent = stageLabel(globalStage);

  for (let idx = 0; idx < nodes.length; idx += 1) {
    const n = nodes[idx];
    const e = THREE.MathUtils.clamp(n.energy, 0, 1);

    let gradAcc = 0;
    let deg = 0;
    let tumorNeighbor = 0;
    for (const key of n.neighbors) {
      const [ia, ib] = key.split("-").map(Number);
      const otherIdx = ia === idx ? ib : ia;
      const other = nodes[otherIdx];
      if (!other) continue;
      gradAcc += Math.abs(n.energy - other.energy);
      deg += 1;
      if (other.type === "tumor") tumorNeighbor = 1;
    }
    const localGrad = deg > 0 ? gradAcc / deg : 0;
    const selfTumor = n.type === "tumor" ? 1 : 0;
    const cellSpreadRisk = THREE.MathUtils.clamp(
      0.56 * (n.energy / 1.2) + 0.32 * localGrad + 0.12 * Math.max(selfTumor, tumorNeighbor),
      0,
      1
    );

    const baseTone = energyToTone(n.type, e);
    const riskTone = new THREE.Color().setHSL(0.01, 0.95, 0.54);
    const blend = THREE.MathUtils.clamp((cellSpreadRisk - spreadThreshold) * 2.1, 0, 1);
    const cellStage = cellSpreadRisk > 0.86 ? 4 : cellSpreadRisk > 0.68 ? 3 : cellSpreadRisk > 0.5 ? 2 : cellSpreadRisk > 0.3 ? 1 : 0;
    n.stage = Math.max(cellStage, Math.max(0, globalStage - 1));
    n.membrane.material.color.copy(baseTone).lerp(riskTone, blend);
    n.membrane.material.opacity = 0.3 + 0.65 * blend + 0.12 * (1 - e);
    n.membrane.material.emissive.copy(riskTone).multiplyScalar(0.1 + 0.9 * blend);

    const pulse = 1 + Math.sin(time * 0.003 + n.velocityPhase) * 0.025;
    n.root.scale.setScalar((0.94 + e * 0.12 + blend * 0.18) * pulse);

    const wiggle = 0.06 + e * 0.08;
    const t = time * 0.001 + n.velocityPhase;
    const invadeDepth = layerDepthForStage(n.stage);
    n.basePosition.copy(n.surfacePosition).add(n.inwardDir.clone().multiplyScalar(invadeDepth));
    n.root.position.x = n.basePosition.x + Math.cos(t) * wiggle;
    n.root.position.y = n.basePosition.y + Math.sin(t * 1.1) * wiggle;
    n.root.position.z = n.basePosition.z + Math.sin(t * 0.9) * wiggle;

    n.nucleus.position.x = Math.cos(t * 1.2) * 0.04;
    n.nucleus.position.y = Math.sin(t * 0.8) * 0.04;

    for (let i = 0; i < n.organelles.length; i += 1) {
      const o = n.organelles[i];
      o.rotation.x += 0.01 + 0.003 * i;
      o.rotation.y += 0.008 + 0.002 * i;
    }

    if (n.type === "tumor") {
      const lesionTone = new THREE.Color().lerpColors(
        new THREE.Color("#cf8a74"),
        new THREE.Color("#b94747"),
        THREE.MathUtils.clamp(cellSpreadRisk, 0, 1)
      );
      n.membrane.material.color.copy(lesionTone);
      n.membrane.material.emissive.copy(new THREE.Color("#4a1515")).multiplyScalar(0.14 + 0.35 * blend);

      if (Array.isArray(n.lesionLobes)) {
        for (let i = 0; i < n.lesionLobes.length; i += 1) {
          const l = n.lesionLobes[i];
          const wob = 1 + 0.06 * Math.sin(t * 1.4 + i);
          l.scale.set((1.0 + 0.22 * (i % 2)) * wob, (0.8 + 0.18 * ((i + 1) % 2)) * wob, 1.0 * wob);
          l.material.color.copy(lesionTone).offsetHSL(0.01 * i, 0.03, -0.04 * i);
          l.material.emissive = new THREE.Color("#5c1a1a");
          l.material.emissiveIntensity = 0.08 + 0.24 * blend;
        }
      }
      if (n.ulcerMesh) {
        const ulcerSeverity = THREE.MathUtils.clamp((cellSpreadRisk - 0.58) * 2.3, 0, 1);
        n.ulcerMesh.visible = ulcerSeverity > 0.05;
        n.ulcerMesh.material.color.lerpColors(
          new THREE.Color("#8a2d2d"),
          new THREE.Color("#5a1111"),
          ulcerSeverity
        );
        n.ulcerMesh.material.emissiveIntensity = 0.1 + 0.28 * ulcerSeverity;
        n.ulcerMesh.scale.set(1.2 + 0.55 * ulcerSeverity, 0.6 + 0.2 * ulcerSeverity, 0.35 + 0.14 * ulcerSeverity);
      }
    }
  }

  for (const line of edgeGroup.children) {
    const { a, b } = line.userData;
    const pa = nodes[a].root.position;
    const pb = nodes[b].root.position;
    const arr = line.geometry.attributes.position.array;
    arr[0] = pa.x; arr[1] = pa.y; arr[2] = pa.z;
    arr[3] = pb.x; arr[4] = pb.y; arr[5] = pb.z;
    line.geometry.attributes.position.needsUpdate = true;

    const ea = THREE.MathUtils.clamp(nodes[a].energy, 0, 1.2);
    const eb = THREE.MathUtils.clamp(nodes[b].energy, 0, 1.2);
    const high = Math.max(ea, eb) / 1.2;
    const gradient = Math.abs(ea - eb);
    const tumorBoost = nodes[a].type === "tumor" || nodes[b].type === "tumor" ? 0.22 : 0;
    const spreadRisk = THREE.MathUtils.clamp(0.52 * high + 0.38 * gradient + tumorBoost, 0, 1);
    line.material.color.setHSL(0.66 * (1 - spreadRisk), 0.95, 0.54 - 0.16 * spreadRisk);
    line.material.opacity = 0.12 + spreadRisk * 0.56;
    line.visible = false;
  }

  for (let i = 0; i < lymphNodes.length; i += 1) {
    const ln = lymphNodes[i];
    const active = globalStage >= 3;
    ln.visible = active;
    if (!active) continue;
    const pulse = 0.82 + 0.32 * Math.sin(time * 0.004 + i);
    ln.scale.setScalar(pulse);
    ln.material.emissiveIntensity = 0.15 + 0.45 * Math.max(0, (globalStage - 2) / 2);
  }

  for (let i = 0; i < metastasisNodes.length; i += 1) {
    const mn = metastasisNodes[i];
    const active = globalStage >= 4;
    mn.visible = active;
    if (!active) continue;
    const pulse = 0.9 + 0.28 * Math.sin(time * 0.003 + i * 0.7);
    mn.scale.setScalar(pulse);
    mn.material.emissiveIntensity = 0.2 + 0.45 * Math.sin(time * 0.002 + i) * 0.5 + 0.25;
  }

  for (let i = 0; i < moistureStreaks.length; i += 1) {
    const m = moistureStreaks[i];
    const pulse = 0.45 + 0.35 * Math.sin(time * 0.003 + m.phase);
    m.mesh.material.opacity = 0.18 + 0.34 * pulse;
    m.mesh.material.emissive = new THREE.Color("#fff1f4");
    m.mesh.material.emissiveIntensity = 0.04 + 0.2 * pulse;
    const s = 0.88 + 0.25 * pulse;
    m.mesh.scale.set(1.9 * s, 0.55 * s, 0.55 * s);
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function captureFramePng(name = `frame-step-${String(simulationStep).padStart(4, "0")}.png`) {
  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, name);
  }, "image/png");
}

function startRecordingVideo() {
  if (isRecording) return;
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
  recordedChunks = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    downloadBlob(blob, `colorectal-sim-${Date.now()}.webm`);
    isRecording = false;
    modeEl.textContent = `${modeLabel} | REC off`;
  };
  mediaRecorder.start();
  isRecording = true;
  modeEl.textContent = `${modeLabel} | REC on`;
}

function stopRecordingVideo() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
}

function recordTreatmentCycle() {
  if (cycleRecording) return;
  cycleTargetSteps = useDatasetPlayback && dataset ? dataset.steps.length : 160;
  resetSimulation();
  startRecordingVideo();
  cycleRecording = true;
}

function tick(time) {
  if (running && time - lastStepAtMs > 260) {
    lastStepAtMs = time;

    if (useDatasetPlayback && dataset && dataset.steps.length > 0) {
      const stepData = dataset.steps[datasetStepIndex % dataset.steps.length];
      datasetStepIndex += 1;
      applyDatasetStep(stepData);
    } else {
      const action = Math.floor(Math.random() * 3);
      applyAction(action);
      simulationStep += 1;
      stepEl.textContent = `Step: ${simulationStep}`;
      actionEl.textContent = `Action: ${action}`;
      avgEnergyEl.textContent = `Avg Energy: ${avgEnergy().toFixed(3)}`;
    }

    progressedSteps += 1;
    updateLesionOverlay();

    if (cycleRecording && progressedSteps >= cycleTargetSteps) {
      cycleRecording = false;
      stopRecordingVideo();
    }
  }

  const dtMs = time - (tick.lastTime || time);
  tick.lastTime = time;
  const dtSec = Math.min(0.05, Math.max(0.001, dtMs * 0.001));
  updateCameraMotion(dtSec);

  updateVisuals(time);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

function resetSimulation() {
  globalStage = 0;
  stageEl.textContent = stageLabel(globalStage);
  simulationStep = 0;
  datasetStepIndex = 0;
  progressedSteps = 0;

  if (useDatasetPlayback && dataset && dataset.steps.length > 0) {
    applyDatasetStep(dataset.steps[0]);
    updateLesionOverlay();
    return;
  }

  for (const n of nodes) n.energy = Math.random() * 0.8 + 0.2;
  stepEl.textContent = "Step: 0";
  actionEl.textContent = "Action: -";
  avgEnergyEl.textContent = `Avg Energy: ${avgEnergy().toFixed(3)}`;
  updateLesionOverlay();
}

function normalizeDataset(raw) {
  if (!raw || typeof raw !== "object") return null;
  if (!Array.isArray(raw.nodes) || raw.nodes.length === 0) return null;
  if (!Array.isArray(raw.steps) || raw.steps.length === 0) return null;

  const cleaned = {
    nodes: raw.nodes.map((n, i) => ({
      id: typeof n.id === "number" ? n.id : i,
      position: Array.isArray(n.position) ? n.position : null,
      energy: typeof n.energy === "number" ? n.energy : 0.5,
      cellType: typeof n.cellType === "string" ? n.cellType : null,
    })),
    edges: Array.isArray(raw.edges) ? raw.edges : [],
    steps: raw.steps
      .map((s, i) => ({
        step: typeof s.step === "number" ? s.step : i,
        action: typeof s.action === "number" ? s.action : null,
        energies: Array.isArray(s.energies) ? s.energies : [],
      }))
      .filter((s) => s.energies.length > 0),
  };

  if (cleaned.nodes.length === 0 || cleaned.steps.length === 0) return null;
  return cleaned;
}

function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("#"));
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",").map((c) => c.trim());
    const row = {};
    for (let i = 0; i < headers.length; i += 1) row[headers[i]] = cols[i] ?? "";
    return row;
  });
}

function toNumber(v, fallback = null) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeType(t) {
  const x = String(t || "").toLowerCase();
  if (x.includes("goblet")) return "goblet";
  if (x.includes("stem")) return "stem";
  if (x.includes("tumor") || x.includes("malig") || x.includes("cancer")) return "tumor";
  return "colonocyte";
}

function buildDatasetFromRealTables(cellsRows, contactRows) {
  if (!Array.isArray(cellsRows) || cellsRows.length === 0) return null;
  const idToIdx = new Map();
  const nodes = [];
  const energies0 = [];
  const energies1 = [];

  for (let i = 0; i < cellsRows.length; i += 1) {
    const r = cellsRows[i];
    const cellId = r.cell_id || r.id || String(i);
    const x = toNumber(r.x, null);
    const y = toNumber(r.y, null);
    const z = toNumber(r.z, 0);
    if (x === null || y === null) continue;

    const energy = toNumber(r.energy, toNumber(r.risk_score, 0.5));
    const post = toNumber(r.energy_post, Math.max(0, energy - 0.08));
    idToIdx.set(cellId, nodes.length);
    nodes.push({
      id: nodes.length,
      position: [x, y, z],
      energy,
      cellType: normalizeType(r.cell_type || r.type),
    });
    energies0.push(energy);
    energies1.push(post);
  }
  if (nodes.length === 0) return null;

  const edges = [];
  for (const c of contactRows || []) {
    const aId = c.src_id || c.source || c.a || c.cell_a;
    const bId = c.dst_id || c.target || c.b || c.cell_b;
    if (!idToIdx.has(aId) || !idToIdx.has(bId)) continue;
    const a = idToIdx.get(aId);
    const b = idToIdx.get(bId);
    if (a === b) continue;
    edges.push([a, b]);
  }

  return normalizeDataset({
    nodes,
    edges,
    steps: [
      { step: 0, action: 0, energies: energies0 },
      { step: 1, action: 1, energies: energies1 },
    ],
  });
}

function mapNodesToColorectum(nodesIn) {
  if (!Array.isArray(nodesIn) || nodesIn.length === 0) return nodesIn;

  const xs = nodesIn.map((n) => Number(n.position?.[0] ?? 0));
  const ys = nodesIn.map((n) => Number(n.position?.[1] ?? 0));
  const zs = nodesIn.map((n) => Number(n.position?.[2] ?? 0));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const spanZ = Math.max(1e-6, maxZ - minZ);

  return nodesIn.map((n, i) => {
    const x = Number(n.position?.[0] ?? i);
    const y = Number(n.position?.[1] ?? i);
    const z = Number(n.position?.[2] ?? 0);
    const tx = (x - minX) / spanX;
    const ty = (y - minY) / spanY;
    const tz = (z - minZ) / spanZ;

    const t = THREE.MathUtils.clamp(0.04 + tx * 0.92, 0.02, 0.98);
    const frame = baseFrames(t);
    const angle = ty * Math.PI * 2;
    const radial = frame.normal
      .clone()
      .multiplyScalar(Math.cos(angle))
      .add(frame.binormal.clone().multiplyScalar(Math.sin(angle)))
      .normalize();
    const wallOffset = (tz - 0.5) * 0.22;
    const r = tubeRadius - 0.25 + wallOffset;
    const mapped = frame.point.clone().add(radial.multiplyScalar(r));

    return {
      ...n,
      position: [mapped.x, mapped.y, mapped.z],
    };
  });
}

async function tryFetchText(path) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

async function loadRealDataset() {
  const cellsCsv = await tryFetchText("./data/real_cells.csv");
  const contactsCsv = await tryFetchText("./data/real_contacts.csv");
  if (!cellsCsv) return null;
  const cellsRows = parseCsv(cellsCsv);
  const contactRows = contactsCsv ? parseCsv(contactsCsv) : [];
  return buildDatasetFromRealTables(cellsRows, contactRows);
}

async function loadDataset() {
  const real = await loadRealDataset();
  if (real) {
    real.nodes = mapNodesToColorectum(real.nodes);
    return { dataset: real, source: "real-spatial" };
  }

  try {
    const res = await fetch("./data/simulation.json", { cache: "no-store" });
    if (!res.ok) return { dataset: null, source: "synthetic" };
    const sim = normalizeDataset(await res.json());
    if (sim) sim.nodes = mapNodesToColorectum(sim.nodes);
    return { dataset: sim, source: "simulation-json" };
  } catch {
    return { dataset: null, source: "synthetic" };
  }
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

window.addEventListener("keydown", (event) => {
  const k = event.key.toLowerCase();
  if (event.key === "Escape") {
    if (helpOpen) {
      helpModal.classList.add("hidden");
      helpModal.setAttribute("aria-hidden", "true");
      helpOpen = false;
      return;
    }
    pressedKeys.clear();
    setCameraPreset("overview");
    return;
  }
  if (helpOpen) return;
  if (keyboardWalkEnabled && ["arrowup", "arrowdown", "arrowleft", "arrowright", "pageup", "pagedown", "w", "a", "s", "d"].includes(k)) {
    pressedKeys.add(k);
    event.preventDefault();
  }
  if (event.code === "Space") running = !running;
  if (k === "r") resetSimulation();
  if (k === "1") setCameraPreset("overview");
  if (k === "2") setCameraPreset("lumen");
  if (k === "3") setCameraPreset("rectum");
  if (k === "p") captureFramePng();
  if (k === "v") {
    if (isRecording) stopRecordingVideo();
    else startRecordingVideo();
  }
  if (k === "b") recordTreatmentCycle();
});

window.addEventListener("keyup", (event) => {
  const k = event.key.toLowerCase();
  if (pressedKeys.has(k)) {
    pressedKeys.delete(k);
    event.preventDefault();
  }
});

controls.addEventListener("start", () => {
  userIsDragging = true;
  cameraAutoBlend = false;
});

controls.addEventListener("end", () => {
  userIsDragging = false;
});

riskThresholdEl.addEventListener("input", (event) => {
  spreadThreshold = Number(event.target.value);
  riskThresholdValueEl.textContent = spreadThreshold.toFixed(2);
  updateLesionOverlay();
});

helpBtn.addEventListener("click", () => {
  helpOpen = true;
  helpModal.classList.remove("hidden");
  helpModal.setAttribute("aria-hidden", "false");
});

closeHelpBtn.addEventListener("click", () => {
  helpOpen = false;
  helpModal.classList.add("hidden");
  helpModal.setAttribute("aria-hidden", "true");
});

helpModal.addEventListener("click", (event) => {
  if (event.target === helpModal) {
    helpOpen = false;
    helpModal.classList.add("hidden");
    helpModal.setAttribute("aria-hidden", "true");
  }
});

helpCard.addEventListener("click", (event) => {
  event.stopPropagation();
});

async function bootstrap() {
  buildColorectalAnatomy();
  buildStageAnatomyTargets();
  initLesionOverlay();

  const loaded = await loadDataset();
  dataset = loaded.dataset;
  datasetSource = loaded.source;
  useDatasetPlayback = dataset !== null;
  if (loaded.source === "real-spatial" && dataset) modeEl.textContent = "Mode: real spatial contacts (mapped to colorectum)";
  else if (loaded.source === "simulation-json" && dataset) modeEl.textContent = "Mode: dataset playback";
  else modeEl.textContent = "Mode: synthetic fallback";
  modeLabel = modeEl.textContent;
  riskThresholdValueEl.textContent = spreadThreshold.toFixed(2);
  riskThresholdEl.value = String(spreadThreshold);

  clearSceneGraph();
  if (useDatasetPlayback) {
    initCellsFromSpec(dataset.nodes);
    initEdgesFromSpec(dataset.edges);
  } else {
    initCellsFromSpec();
    initEdgesFromSpec();
  }

  setCameraPreset("overview", true);
  resetSimulation();
  requestAnimationFrame(tick);
}

bootstrap();

