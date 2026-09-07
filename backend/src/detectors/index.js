import { RoboflowSnookerDetector } from './RoboflowSnookerDetector.js';
import { HFDetector } from './HFDetector.js';

/** Registry of interchangeable pretrained / future detectors. */
const DETECTORS = {
  'roboflow-snooker': () => new RoboflowSnookerDetector(),
  'hf-space': () => new HFDetector(),
};

export function listDetectors() {
  return Object.keys(DETECTORS).map((id) => {
    const det = DETECTORS[id]();
    return { id, ...det.getModelInfo() };
  });
}

export function getDetector(id = 'roboflow-snooker') {
  const factory = DETECTORS[id] || DETECTORS['roboflow-snooker'];
  if (!factory) {
    const err = new Error(`Unknown detector: ${id}`);
    err.status = 404;
    throw err;
  }
  return factory();
}

export { RoboflowSnookerDetector, HFDetector };
