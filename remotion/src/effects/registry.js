/**
 * Effect registry for the Promo composition.
 *
 * Each effect is a pure function that maps (frame, durationInFrames) → a
 * style fragment.  Multiple effects are merged: motion effects contribute
 * to `transform`, lighting to `filter`, opacity effects to `opacity`, and
 * overlay effects return a child element rendered on top of the region.
 *
 * Registry shape:
 *   { [key]: { kind, compute } }
 *
 * compute(frame, durationInFrames) returns one of:
 *   - { transform }          for motion effects
 *   - { filter }             for lighting effects
 *   - { opacity }            for opacity effects
 */

import { interpolate, Easing } from "remotion";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Normalised playhead 0..1 */
const progress = (frame, dur) => frame / Math.max(dur, 1);

/** A smooth sin-wave ping-pong from 0→1→0 over the full clip */
const wave = (frame, dur) =>
  interpolate(frame, [0, dur / 2, dur], [0, 1, 0], {
    easing: Easing.inOut(Easing.sin),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

/** Full sin-period 0→1→0→-1→0 over the clip */
const fullSin = (frame, dur) =>
  Math.sin((frame / Math.max(dur, 1)) * Math.PI * 2);

// ---------------------------------------------------------------------------
// Effect definitions
// ---------------------------------------------------------------------------

const REGISTRY = {
  // -------------------------------------------------------------------------
  // MOTION
  // -------------------------------------------------------------------------

  float: {
    kind: "motion",
    compute(frame, dur) {
      // One full sin-wave cycle over the clip → perfectly smooth loop
      const ty = Math.sin((frame / Math.max(dur, 1)) * Math.PI * 2) * -10;
      return { transform: `translateY(${ty}px)` };
    },
  },

  breathe: {
    kind: "motion",
    compute(frame, dur) {
      const s = 1 + interpolate(wave(frame, dur), [0, 1], [0, 0.015], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { transform: `scale(${s})` };
    },
  },

  "natural-breathe": {
    kind: "motion",
    compute(frame, dur) {
      // Scales only on Y axis from the bottom. Mimics taking a breath.
      // Zero horizontal expansion means it will NEVER overlap adjacent UI cards!
      const sy = 1 + interpolate(wave(frame, dur), [0, 1], [0, 0.025], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { 
        transform: `scaleY(${sy})`,
        transformOrigin: "bottom center" 
      };
    },
  },



  zoom: {
    kind: "motion",
    compute(frame, dur) {
      const s = 1 + interpolate(wave(frame, dur), [0, 1], [0, 0.03], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { transform: `scale(${s})` };
    },
  },

  bounce: {
    kind: "motion",
    compute(frame, dur) {
      // Sharp parabolic bounce – up then snappy fall
      const p = progress(frame, dur);
      const cycle = p % (1 / 3); // 3 bounces per clip
      const phase = cycle * 3;
      const ty = -Math.abs(Math.sin(phase * Math.PI)) * 10;
      return { transform: `translateY(${ty}px)` };
    },
  },

  shake: {
    kind: "motion",
    compute(frame) {
      // Rapid horizontal micro-jitter
      const tx = Math.sin(frame * 1.7) * 3 + Math.sin(frame * 3.1) * 1.5;
      return { transform: `translateX(${tx}px)` };
    },
  },

  wave: {
    kind: "motion",
    compute(frame, dur) {
      const tx = interpolate(
        progress(frame, dur),
        [0, 0.25, 0.75, 1],
        [0, 8, -8, 0],
        { easing: Easing.inOut(Easing.sin), extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      );
      return { transform: `translateX(${tx}px)` };
    },
  },

  spin: {
    kind: "motion",
    compute(frame, dur) {
      const deg = progress(frame, dur) * 360;
      return { transform: `rotate(${deg}deg)` };
    },
  },

  // -------------------------------------------------------------------------
  // LIGHTING
  // -------------------------------------------------------------------------

  glow: {
    kind: "filter",
    compute(frame, dur) {
      const intensity = interpolate(wave(frame, dur), [0, 1], [4, 14], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { filter: `drop-shadow(0 0 ${intensity}px rgba(100,255,100,0.85))` };
    },
  },

  rim: {
    kind: "filter",
    compute(frame, dur) {
      const w = wave(frame, dur);
      const intensity = interpolate(w, [0, 1], [2, 8], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      const opacity = interpolate(w, [0, 1], [0.4, 1.0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { filter: `drop-shadow(0 0 ${intensity}px rgba(180,140,255,${opacity}))` };
    },
  },

  neon_pulse: {
    kind: "filter",
    compute(frame, dur, regionColor) {
      const w = wave(frame, dur);
      // Pulsing intense glow filter 
      const blur = 5 + w * 15;
      const opacity = 0.5 + w * 0.5;
      const c = regionColor || "rgba(0, 255, 200, 1)";
      return { filter: `drop-shadow(0px 0px ${blur}px ${c}) opacity(${opacity})` };
    },
  },

  gold_pulse: {
    kind: "filter",
    compute(frame, dur) {
      const w = wave(frame, dur);
      const sepia = interpolate(w, [0, 1], [0.1, 0.55], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      const saturate = interpolate(w, [0, 1], [1, 1.8], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return { filter: `sepia(${sepia}) saturate(${saturate})` };
    },
  },

  shine: {
    kind: "filter",
    compute(frame, dur) {
      const p = progress(frame, dur);
      const brightness = 1 + interpolate(
        p,
        [0, 0.35, 0.65, 1],
        [0, 0.35, 0, 0],
        { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      );
      return { filter: `brightness(${brightness})` };
    },
  },

  flicker: {
    kind: "opacity",
    compute(frame) {
      // Neon-sign rapid random flicker
      const seed = (Math.sin(frame * 7.3 + 1.1) + 1) / 2;
      return { opacity: 0.75 + seed * 0.25 };
    },
  },

  fade: {
    kind: "opacity",
    compute(frame, dur) {
      const op = interpolate(
        progress(frame, dur),
        [0, 0.3, 0.7, 1],
        [0.4, 1, 1, 0.4],
        { easing: Easing.inOut(Easing.sin), extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      );
      return { opacity: op };
    },
  },

  // -------------------------------------------------------------------------
  // PARTICLE / COLOUR
  // -------------------------------------------------------------------------

  sparkle: {
    kind: "lottie",
    // Rendered separately via <LottieOverlay> using the existing sparkle.json
  },

  color_shift: {
    kind: "filter",
    compute(frame, dur) {
      const deg = progress(frame, dur) * 60; // 0..60° hue shift
      return { filter: `hue-rotate(${deg}deg)` };
    },
  },
};

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/**
 * Compute merged CSS style for all *non-lottie* effects on a region.
 *
 * @param {string[]} effects   - effect keys active on this region
 * @param {number}   frame     - current Remotion frame
 * @param {number}   dur       - total durationInFrames
 * @param {string}   [regionColor] - optional custom hex color from UI
 * @returns {React.CSSProperties}
 */
export function computeEffectStyle(effects, frame, dur, regionColor) {
  const t = progress(frame, dur);
  let transformStr = "";
  let filterStr = "";
  let style = { opacity: 1 };

  // Float & Glow
  if (effects.includes("float-glow")) {
    const yOffset = Math.sin(t * Math.PI * 2) * 15;
    transformStr += ` translateY(${yOffset}px)`;
    const blur = 10 + Math.sin(t * Math.PI * 2) * 5;
    const c = regionColor || "rgba(120, 200, 255, 0.8)";
    filterStr += ` drop-shadow(0 0 ${blur}px ${c})`;
  }

  // Slide Left
  if (effects.includes("slide-left")) {
    const slideProgress = Math.min(1, t * 3);
    const x = (1 - slideProgress) * -200;
    transformStr += ` translateX(${x}px)`;
    style.opacity = Math.min(1, t * 5);
  }

  // Elastic Zoom-in
  if (effects.includes("zoom-in")) {
    const p = t * 3;
    let s = 1;
    if (p < 1) {
      // Elastic out formula
      s = 1 - Math.cos(p * Math.PI * 4.5) * Math.exp(-p * 6);
    }
    transformStr += ` scale(${s})`;
  }

  // Impact Shake
  if (effects.includes("shake")) {
    if (t < 0.2) {
      const shakeAmt = (0.2 - t) * 100;
      const x = Math.sin(t * 100) * shakeAmt;
      const y = Math.cos(t * 110) * shakeAmt;
      transformStr += ` translate(${x}px, ${y}px)`;
    }
  }

  // Rainbow Cycle
  if (effects.includes("rainbow")) {
    const hue = Math.floor((t * 360 * 2) % 360);
    filterStr += ` hue-rotate(${hue}deg) saturate(1.5)`;
  }

  const transforms = [];
  const filters = [];
  let opacity = style.opacity;

  for (const key of effects) {
    const entry = REGISTRY[key];
    if (!entry || entry.kind === "lottie" || ["float-glow", "slide-left", "zoom-in", "shake", "rainbow"].includes(key)) {
      continue;
    }
    const result = entry.compute(frame, dur, regionColor);
    if (result.transform) transforms.push(result.transform);
    if (result.filter) filters.push(result.filter);
    if (result.opacity !== undefined) opacity *= result.opacity;
  }

  if (transformStr) transforms.push(transformStr.trim());
  if (filterStr) filters.push(filterStr.trim());

  return {
    ...(transforms.length ? { transform: transforms.join(" ") } : {}),
    ...(filters.length ? { filter: filters.join(" ") } : {}),
    opacity,
  };
}

/**
 * Motion for a SAM character cut-out.
 *
 * Uniform scale() on a person PNG stretches the face (eyes/mouth drift).
 * Characters only get rigid translation; lighting/opacity still apply.
 * Zoom is skipped on purpose — it is a card effect, not a head warp.
 */
export function computeCharacterEffectStyle(effects, frame, dur) {
  const t = progress(frame, dur);
  const w = wave(frame, dur);
  const transforms = [];
  const filters = [];
  let opacity = 1;
  let ty = 0;
  let tx = 0;

  if (effects.includes("float") || effects.includes("float-glow")) {
    ty += interpolate(t, [0, 0.5, 1], [0, -5, 0], {
      easing: Easing.inOut(Easing.sin),
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }
  if (effects.includes("breathe")) {
    ty += interpolate(w, [0, 1], [0, -2.5], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }
  if (effects.includes("bounce")) {
    const cycle = t % (1 / 3);
    ty += -Math.abs(Math.sin(cycle * 3 * Math.PI)) * 8;
  }
  if (effects.includes("wave")) {
    tx += interpolate(t, [0, 0.25, 0.75, 1], [0, 6, -6, 0], {
      easing: Easing.inOut(Easing.sin),
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }
  if (effects.includes("shake") && t < 0.2) {
    const shakeAmt = (0.2 - t) * 40;
    tx += Math.sin(t * 100) * shakeAmt;
    ty += Math.cos(t * 110) * shakeAmt;
  }
  if (tx || ty) {
    transforms.push(`translate(${tx}px, ${ty}px)`);
  }

  if (effects.includes("float-glow")) {
    const blur = 10 + Math.sin(t * Math.PI * 2) * 5;
    filters.push(`drop-shadow(0 0 ${blur}px rgba(120, 200, 255, 0.8))`);
  }
  if (effects.includes("rainbow")) {
    const hue = Math.floor((t * 360 * 2) % 360);
    filters.push(`hue-rotate(${hue}deg) saturate(1.5)`);
  }

  const lightingSkip = new Set([
    "float", "float-glow", "breathe", "zoom", "zoom-in", "bounce",
    "shake", "wave", "spin", "slide-left", "parallax", "sparkle",
  ]);
  for (const key of effects) {
    if (lightingSkip.has(key)) continue;
    const entry = REGISTRY[key];
    if (!entry || entry.kind === "lottie") continue;
    const result = entry.compute(frame, dur);
    if (result.filter) filters.push(result.filter);
    if (result.opacity !== undefined) opacity *= result.opacity;
  }

  return {
    ...(transforms.length ? { transform: transforms.join(" ") } : {}),
    ...(filters.length ? { filter: filters.join(" ") } : {}),
    opacity,
  };
}

/**
 * Returns true if any of the given effect keys need a Lottie overlay.
 *
 * @param {string[]} effects
 * @returns {boolean}
 */
export function needsLottie(effects) {
  return effects.some((k) => REGISTRY[k]?.kind === "lottie");
}

export { REGISTRY };
