import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';

const CLASS_COLORS = {
  red: '#e74c3c',
  yellow: '#f1c40f',
  green: '#27ae60',
  brown: '#a0522d',
  blue: '#3498db',
  pink: '#ff69b4',
  black: '#111111',
  cue_ball: '#ecf0f1',
};

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

/**
 * Manual bounding-box annotator for SCOS dataset images.
 * No auto-labeling — boxes are drawn/edited by hand.
 */
export default function DatasetAnnotator({ onBack }) {
  const [images, setImages] = useState([]);
  const [index, setIndex] = useState(0);
  const [classes, setClasses] = useState([]);
  const [boxes, setBoxes] = useState([]);
  const [activeClass, setActiveClass] = useState('red');
  const [selected, setSelected] = useState(-1);
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const stageRef = useRef(null);
  const imgRef = useRef(null);
  const dragRef = useRef(null);
  const spaceRef = useRef(false);

  const current = images[index] || null;

  const reloadList = useCallback(async (preferId) => {
    const [list, cls, st] = await Promise.all([
      api.datasetImages({ limit: 2000 }),
      api.datasetClasses(),
      api.datasetStats(),
    ]);
    setImages(list.images || []);
    setClasses(cls.classes || []);
    setStats(st);
    if (preferId) {
      const i = (list.images || []).findIndex((r) => r.image_id === preferId);
      if (i >= 0) setIndex(i);
    }
  }, []);

  useEffect(() => {
    reloadList().catch((e) => setError(e.message));
  }, [reloadList]);

  useEffect(() => {
    if (!current) {
      setBoxes([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const data = await api.datasetGetAnnotation(current.image_id);
        if (!cancelled) {
          setBoxes(data.annotation?.boxes || []);
          setSelected(-1);
          setError('');
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [current?.image_id]);

  useEffect(() => {
    function onKey(e) {
      if (e.code === 'Space') spaceRef.current = e.type === 'keydown';
      if (e.type !== 'keydown') return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selected >= 0) {
          setBoxes((prev) => prev.filter((_, i) => i !== selected));
          setSelected(-1);
        }
      }
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(i + 1, Math.max(0, images.length - 1)));
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(0, i - 1));
    }
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
    };
  }, [selected, images.length]);

  function toImagePoint(evt) {
    const img = imgRef.current;
    if (!img || !imgSize.w) return null;
    const rect = img.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * imgSize.w;
    const y = ((evt.clientY - rect.top) / rect.height) * imgSize.h;
    return { x: clamp(x, 0, imgSize.w), y: clamp(y, 0, imgSize.h) };
  }

  function hitTest(p) {
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
      const b = boxes[i];
      if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2) return i;
    }
    return -1;
  }

  function handlePointerDown(evt) {
    const p = toImagePoint(evt);
    if (!p) return;
    evt.currentTarget.setPointerCapture?.(evt.pointerId);

    if (spaceRef.current || evt.button === 1) {
      dragRef.current = { mode: 'pan', ox: evt.clientX - pan.x, oy: evt.clientY - pan.y };
      return;
    }

    const hit = hitTest(p);
    if (hit >= 0) {
      const b = boxes[hit];
      const nearR = Math.abs(p.x - b.x2) < 12 && Math.abs(p.y - b.y2) < 12;
      setSelected(hit);
      dragRef.current = nearR
        ? { mode: 'resize', index: hit, start: p, box: { ...b } }
        : {
            mode: 'move',
            index: hit,
            start: p,
            box: { ...b },
          };
      return;
    }

    setSelected(-1);
    dragRef.current = {
      mode: 'draw',
      start: p,
      class: activeClass,
    };
    setBoxes((prev) => [
      ...prev,
      { class: activeClass, x1: p.x, y1: p.y, x2: p.x, y2: p.y },
    ]);
    setSelected(boxes.length);
  }

  function handlePointerMove(evt) {
    const drag = dragRef.current;
    if (!drag) return;

    if (drag.mode === 'pan') {
      setPan({ x: evt.clientX - drag.ox, y: evt.clientY - drag.oy });
      return;
    }

    const p = toImagePoint(evt);
    if (!p) return;

    if (drag.mode === 'draw') {
      setBoxes((prev) => {
        const next = [...prev];
        const i = next.length - 1;
        next[i] = {
          ...next[i],
          x2: p.x,
          y2: p.y,
        };
        return next;
      });
      return;
    }

    if (drag.mode === 'move') {
      const dx = p.x - drag.start.x;
      const dy = p.y - drag.start.y;
      setBoxes((prev) => {
        const next = [...prev];
        const b = drag.box;
        let x1 = b.x1 + dx;
        let y1 = b.y1 + dy;
        let x2 = b.x2 + dx;
        let y2 = b.y2 + dy;
        const w = x2 - x1;
        const h = y2 - y1;
        x1 = clamp(x1, 0, imgSize.w - w);
        y1 = clamp(y1, 0, imgSize.h - h);
        next[drag.index] = { ...b, x1, y1, x2: x1 + w, y2: y1 + h };
        return next;
      });
      return;
    }

    if (drag.mode === 'resize') {
      setBoxes((prev) => {
        const next = [...prev];
        const b = drag.box;
        next[drag.index] = {
          ...b,
          x2: clamp(p.x, b.x1 + 4, imgSize.w),
          y2: clamp(p.y, b.y1 + 4, imgSize.h),
        };
        return next;
      });
    }
  }

  function handlePointerUp() {
    const drag = dragRef.current;
    dragRef.current = null;
    if (drag?.mode === 'draw') {
      setBoxes((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (!last) return prev;
        let { x1, y1, x2, y2 } = last;
        if (x2 < x1) [x1, x2] = [x2, x1];
        if (y2 < y1) [y1, y2] = [y2, y1];
        if (x2 - x1 < 4 || y2 - y1 < 4) {
          next.pop();
          setSelected(-1);
          return next;
        }
        next[next.length - 1] = { ...last, x1, y1, x2, y2 };
        return next;
      });
    }
  }

  function onWheel(evt) {
    evt.preventDefault();
    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    setScale((s) => clamp(s * factor, 0.4, 6));
  }

  async function save() {
    if (!current) return;
    setStatus('Saving…');
    try {
      await api.datasetSaveAnnotation(current.image_id, {
        boxes,
        width: imgSize.w,
        height: imgSize.h,
      });
      setStatus(`Saved ${boxes.length} box(es)`);
      await reloadList(current.image_id);
    } catch (err) {
      setError(err.message);
      setStatus('');
    }
  }

  function changeSelectedClass(name) {
    setActiveClass(name);
    if (selected >= 0) {
      setBoxes((prev) => prev.map((b, i) => (i === selected ? { ...b, class: name } : b)));
    }
  }

  const progress = useMemo(() => {
    if (!stats) return '';
    return `${stats.annotated_images}/${stats.total_images} annotated`;
  }, [stats]);

  if (!images.length) {
    return (
      <div className="panel">
        <div className="panel-head">
          <h2>Annotate</h2>
          <button type="button" className="btn ghost" onClick={onBack}>
            Back to capture
          </button>
        </div>
        <p className="empty">No captured images yet — capture frames first.</p>
      </div>
    );
  }

  return (
    <div className="dataset-ann">
      <div className="panel dataset-ann-toolbar">
        <div className="panel-head">
          <h2>Annotate</h2>
          <span>{progress}</span>
        </div>
        <div className="calib-actions">
          <button type="button" className="btn ghost" onClick={onBack}>
            Back to capture
          </button>
          <button
            type="button"
            className="btn ghost"
            disabled={index <= 0}
            onClick={() => setIndex((i) => i - 1)}
          >
            ← Prev
          </button>
          <span className="hint">
            {index + 1} / {images.length}
            {current?.annotated ? ' · annotated' : ' · unannotated'}
          </span>
          <button
            type="button"
            className="btn ghost"
            disabled={index >= images.length - 1}
            onClick={() => setIndex((i) => i + 1)}
          >
            Next →
          </button>
          <button type="button" className="btn primary" onClick={save}>
            Save annotation
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={() => {
              setScale(1);
              setPan({ x: 0, y: 0 });
            }}
          >
            Reset view
          </button>
        </div>
        <p className="hint">
          Draw: click-drag · Move box: drag inside · Resize: drag bottom-right handle · Pan: hold
          Space · Zoom: wheel · Delete: Del · Class: click legend (applies to new / selected box)
        </p>
        {(status || error) && (
          <div className={`banner ${error ? 'bad' : 'ok'}`}>{error || status}</div>
        )}
      </div>

      <div className="dataset-ann-body">
        <aside className="panel dataset-legend">
          <h3>Classes</h3>
          <ul>
            {(classes.length ? classes : Object.keys(CLASS_COLORS).map((name) => ({ name }))).map(
              (c) => (
                <li key={c.name}>
                  <button
                    type="button"
                    className={activeClass === c.name ? 'on' : ''}
                    onClick={() => changeSelectedClass(c.name)}
                  >
                    <span
                      className="swatch"
                      style={{ background: c.color || CLASS_COLORS[c.name] || '#888' }}
                    />
                    {c.name}
                    {stats?.annotations_by_class?.[c.name] != null
                      ? ` · ${stats.annotations_by_class[c.name]}`
                      : ''}
                  </button>
                </li>
              )
            )}
          </ul>
          <h3>Boxes on image</h3>
          <ul className="box-list">
            {boxes.map((b, i) => (
              <li key={i} className={selected === i ? 'on' : ''}>
                <button type="button" onClick={() => setSelected(i)}>
                  {b.class}
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => {
                    setBoxes((prev) => prev.filter((_, j) => j !== i));
                    setSelected(-1);
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <div className="panel dataset-ann-stage-wrap">
          <div
            className="dataset-ann-stage"
            ref={stageRef}
            onWheel={onWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <div
              className="dataset-ann-world"
              style={{
                transform: `translate(calc(-50% + ${pan.x}px), calc(-50% + ${pan.y}px)) scale(${scale})`,
              }}
            >
              {current && (
                <img
                  ref={imgRef}
                  src={api.datasetFileUrl(current.image_id)}
                  alt={current.image_id}
                  draggable={false}
                  onLoad={(e) =>
                    setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })
                  }
                />
              )}
              {imgSize.w > 0 && (
                <svg
                  className="dataset-ann-svg"
                  viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
                  width={imgSize.w}
                  height={imgSize.h}
                >
                  {boxes.map((b, i) => {
                    const color = CLASS_COLORS[b.class] || '#e8b84a';
                    const x = Math.min(b.x1, b.x2);
                    const y = Math.min(b.y1, b.y2);
                    const w = Math.abs(b.x2 - b.x1);
                    const h = Math.abs(b.y2 - b.y1);
                    return (
                      <g key={i} className={selected === i ? 'sel' : ''}>
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={h}
                          fill={`${color}33`}
                          stroke={color}
                          strokeWidth={selected === i ? 3 : 2}
                          vectorEffect="non-scaling-stroke"
                        />
                        <text x={x + 4} y={y - 6} fill={color} fontSize="18" fontWeight="700">
                          {b.class}
                        </text>
                        {selected === i && (
                          <rect
                            x={Math.max(b.x1, b.x2) - 8}
                            y={Math.max(b.y1, b.y2) - 8}
                            width={16}
                            height={16}
                            fill={color}
                            stroke="#fff"
                            strokeWidth="2"
                            vectorEffect="non-scaling-stroke"
                          />
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
