import { useEffect, useRef, useState } from 'react';

let jsmpegPromise = null;

function loadJSMpeg() {
  if (window.JSMpeg) return Promise.resolve(window.JSMpeg);
  if (jsmpegPromise) return jsmpegPromise;

  jsmpegPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/jsmpeg.min.js';
    script.async = true;
    script.onload = () => {
      if (window.JSMpeg) resolve(window.JSMpeg);
      else reject(new Error('JSMpeg not available'));
    };
    script.onerror = () => reject(new Error('Failed to load JSMpeg'));
    document.head.appendChild(script);
  });

  return jsmpegPromise;
}

export default function LivePlayer({ wsUrl, title, onStop }) {
  const canvasRef = useRef(null);
  const playerRef = useRef(null);
  const [status, setStatus] = useState('connecting');
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    let player = null;

    async function start() {
      try {
        setStatus('loading player');
        const JSMpeg = await loadJSMpeg();
        if (cancelled || !canvasRef.current) return;

        setStatus('connecting');
        player = new JSMpeg.Player(wsUrl, {
          canvas: canvasRef.current,
          autoplay: true,
          audio: false,
          disableGl: false,
          preserveDrawingBuffer: true,
          onSourceEstablished: () => {
            if (!cancelled) setStatus('live');
          },
          onSourceCompleted: () => {
            if (!cancelled) setStatus('ended');
          },
        });
        playerRef.current = player;

        // Some builds don't fire callbacks reliably — mark live after short wait if streaming
        setTimeout(() => {
          if (!cancelled && status !== 'ended') setStatus((s) => (s === 'connecting' || s === 'loading player' ? 'live' : s));
        }, 2500);
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setStatus('error');
        }
      }
    }

    start();

    return () => {
      cancelled = true;
      try {
        player?.destroy?.();
      } catch {
        /* ignore */
      }
      playerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsUrl]);

  return (
    <div className="player">
      <div className="player-top">
        <div className="player-title">
          <span className={`live-dot ${status === 'live' ? 'on' : ''}`} />
          <div>
            <strong>{title}</strong>
            <small>{status === 'live' ? 'Live stream' : status}</small>
          </div>
        </div>
        {onStop && (
          <button type="button" className="btn ghost" onClick={onStop}>
            Stop
          </button>
        )}
      </div>
      <div className="player-stage">
        <canvas ref={canvasRef} className="player-canvas" width={960} height={540} />
        {status !== 'live' && !error && (
          <div className="player-overlay">
            <div className="scan" />
            <p>{status}…</p>
          </div>
        )}
        {error && <div className="player-overlay error">{error}</div>}
      </div>
    </div>
  );
}
