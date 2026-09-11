import React from "react";
import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  interpolateColors,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import { computeEffectStyle } from "./effects/registry";

function assetSrc(src) {
  if (!src) return null;
  if (
    src.startsWith("http://") ||
    src.startsWith("https://") ||
    src.startsWith("file:") ||
    src.startsWith("data:")
  ) {
    return src;
  }
  return staticFile(src);
}

const PERSON_SOURCES = new Set(["yolo", "sam"]);
const UI_SOURCES = new Set(["card", "button", "title", "ocr", "prop", "manual"]);
const PIXEL_MOTION = new Set([
  "float", "float-glow", "breathe", "natural-breathe", "zoom", "zoom-in", "pulse",
  "bounce", "shake", "wave", "spin", "slide-left", "slide-up", "tilt",
]);

function isPersonRegion(region) {
  const src = (region.source || "").toLowerCase();
  const label = (region.label || "").toLowerCase();
  return PERSON_SOURCES.has(src) || label.includes("person") || label.includes("character");
}

function isUiRegion(region) {
  const src = (region.source || "").toLowerCase();
  return UI_SOURCES.has(src) && !isPersonRegion(region);
}

function isPlaqueCutout(item) {
  const src = (item.source || "").toLowerCase();
  return src === "card" || src === "button" || src === "title";
}

function hasEffect(effects, name) {
  return Array.isArray(effects) && effects.includes(name);
}

function boxPixels(region, canvasW, canvasH) {
  const left = region.x * canvasW;
  const top = region.y * canvasH;
  const width = region.width * canvasW;
  const height = region.height * canvasH;
  const src = (region.source || "").toLowerCase();
  const radius =
    src === "button"
      ? Math.max(4, height / 2)
      : src === "card"
        ? Math.min(28, Math.min(width, height) * 0.22)
        : Math.min(16, Math.min(width, height) * 0.12);
  return { left, top, width, height, radius };
}

function useLoopWave() {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const duration = Math.max(durationInFrames, 1);
  return interpolate(frame, [0, duration / 2, duration], [0, 1, 0], {
    easing: Easing.inOut(Easing.sin),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

function CornerTwinkle({ region, canvasW, canvasH }) {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const t = frame / Math.max(durationInFrames, 1);
  const pad = Math.min(width, height) * 0.16;
  const star = Math.max(6, Math.min(12, Math.min(width, height) * 0.08));
  const spots = [
    { x: pad, y: pad, phase: 0 },
    { x: width - pad, y: pad, phase: 0.28 },
    { x: pad, y: height - pad, phase: 0.52 },
    { x: width - pad, y: height - pad, phase: 0.76 },
  ];

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        borderRadius: radius,
        pointerEvents: "none",
      }}
    >
      {spots.map((spot, index) => {
        const pulse = 0.2 + 0.8 * (0.5 + 0.5 * Math.sin((t + spot.phase) * Math.PI * 2));
        return (
          <div
            key={index}
            style={{
              position: "absolute",
              left: spot.x - star / 2,
              top: spot.y - star / 2,
              width: star,
              height: star,
              opacity: pulse,
              background:
                "radial-gradient(circle, rgba(255,248,220,0.95) 0%, rgba(255,210,90,0.7) 38%, transparent 70%)",
              mixBlendMode: "screen",
            }}
          />
        );
      })}
    </div>
  );
}

function ShineBand({ region, canvasW, canvasH }) {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const dur = Math.max(durationInFrames, 1);

  // Phase 0-70% of clip: slow cubic ease-in sweep from -20 → 110
  // Phase 70-100%: hold at 110 (band has exited, nothing visible)
  const sweepFrameEnd = Math.floor(dur * 0.70);
  const sweep = interpolate(
    frame,
    [0, sweepFrameEnd],
    [-20, 110],
    {
      easing: Easing.in(Easing.cubic),
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  // Fade the whole band out in the last 15% of the clip so the loop is clean
  const bandOpacity = interpolate(
    frame,
    [Math.floor(dur * 0.60), Math.floor(dur * 0.75)],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        borderRadius: radius,
        pointerEvents: "none",
        mixBlendMode: "screen",
        opacity: bandOpacity,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          // Narrower band (10pp wide) and softer peak (0.55 vs 0.72)
          background: `linear-gradient(115deg, transparent 0%, transparent ${sweep}%, rgba(255, 214, 110, 0) ${sweep}%, rgba(255, 214, 110, 0.42) ${sweep + 5}%, rgba(255, 186, 70, 0.08) ${sweep + 10}%, transparent 100%)`,
        }}
      />
    </div>
  );
}

function SheenBand({ region, canvasW, canvasH }) {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const sweep = interpolate(frame, [0, Math.max(durationInFrames, 1)], [-15, 120], {
    easing: Easing.inOut(Easing.quad),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        borderRadius: radius,
        pointerEvents: "none",
        mixBlendMode: "screen",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(115deg, transparent 0%, transparent ${sweep}%, rgba(255, 255, 255, 0) ${sweep}%, rgba(255, 248, 220, 0.95) ${sweep + 3}%, rgba(255, 210, 90, 0) ${sweep + 8}%, transparent 100%)`,
        }}
      />
    </div>
  );
}

function GlowWash({ region, canvasW, canvasH, color, opacity }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        borderRadius: radius,
        background: color,
        opacity,
        mixBlendMode: "screen",
        pointerEvents: "none",
      }}
    />
  );
}

function RimGlow({ region, canvasW, canvasH, strength }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        borderRadius: radius,
        boxShadow: `inset 0 0 ${8 + 16 * strength}px rgba(255, 214, 110, ${0.2 + 0.4 * strength})`,
        pointerEvents: "none",
      }}
    />
  );
}

function wantsPixelMotion(effects) {
  return (effects || []).some((key) => PIXEL_MOTION.has(key));
}

function OverlayFX({ region, canvasW, canvasH, dur, wave, faceWash = true }) {
  const effects = region.effects || [];
  const glowColor = interpolateColors(wave, [0, 1], ["rgba(255, 236, 180, 0.0)", "rgba(255, 236, 180, 0.32)"]);
  const wash = faceWash && hasEffect(effects, "glow");
  const rim = faceWash && hasEffect(effects, "rim");

  return (
    <>
      {wash ? (
        <GlowWash
          region={region}
          canvasW={canvasW}
          canvasH={canvasH}
          color={region.color || glowColor}
          opacity={1}
        />
      ) : null}
      {hasEffect(effects, "shine") ? (
        <ShineBand region={region} canvasW={canvasW} canvasH={canvasH} />
      ) : null}
      {hasEffect(effects, "sheen") ? (
        <SheenBand region={region} canvasW={canvasW} canvasH={canvasH} />
      ) : null}
      {rim ? (
        <RimGlow region={region} canvasW={canvasW} canvasH={canvasH} strength={wave} color={region.color || glowColor} />
      ) : null}
      {hasEffect(effects, "twinkle") ? (
        <CornerTwinkle region={region} canvasW={canvasW} canvasH={canvasH} />
      ) : null}
    </>
  );
}

function UiLayer({ region, posterSrc, originalSrc, canvasW, canvasH, frame, dur, wave }) {
  const effects = region.effects || [];
  const src = (region.source || "").toLowerCase();
  // card / button / title regions already have a CutoutLayer that moves the
  // whole cut-out PNG as one unit. Running pixel-motion here too causes the
  // poster crop to scale *inside* the static frame — "content moving in box".
  const isPlaqueRegion = src === "card" || src === "button" || src === "title";

  // For OCR text: we use the original (pre-inpaint) image for the zoom crop
  // so we see real pixels rather than the white inpainted background.
  const isOcr = src === "ocr";
  const motion = !isPlaqueRegion && !isOcr && wantsPixelMotion(effects);
  const ocrMotion = isOcr && wantsPixelMotion(effects);
  const effectStyle = (motion || ocrMotion) ? computeEffectStyle(effects, frame, dur) : {};
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  // Which image to use as the zoom crop source:
  // - non-OCR regions: posterSrc (inpainted background, cards already removed)
  // - OCR regions: originalSrc (real image, cards still present behind text)
  const cropSrc = isOcr ? (originalSrc || posterSrc) : posterSrc;

  return (
    <>
      {/* Non-OCR regions: poster-crop inside a zooming clip div */}
      {motion ? (
        <div
          style={{
            position: "absolute",
            left,
            top,
            width,
            height,
            overflow: "hidden",
            borderRadius: radius,
            pointerEvents: "none",
            transformOrigin: "center center",
            ...effectStyle,
          }}
        >
          <Img
            src={cropSrc}
            style={{
              position: "absolute",
              left: -left,
              top: -top,
              width: canvasW,
              height: canvasH,
              objectFit: "fill",
            }}
          />
        </div>
      ) : null}

      {/* plaque regions: shine/glow overlays are handled by CutoutLayer */}
      {!isPlaqueRegion && !isOcr ? (
        <OverlayFX region={region} canvasW={canvasW} canvasH={canvasH} dur={dur} wave={wave} />
      ) : null}

      {/* OCR regions: zoom uses the original (pre-inpaint) image crop so
          text pixels show through, not the white inpainted background. */}
      {isOcr ? (
        <div
          style={{
            position: "absolute",
            left,
            top,
            width,
            height,
            overflow: "hidden",
            borderRadius: radius,
            pointerEvents: "none",
            transformOrigin: "center center",
            ...(ocrMotion ? effectStyle : {}),
          }}
        >
          {ocrMotion ? (
            <Img
              src={cropSrc}
              style={{
                position: "absolute",
                left: -left,
                top: -top,
                width: canvasW,
                height: canvasH,
                objectFit: "fill",
              }}
            />
          ) : null}
          <OverlayFX
            region={{ ...region, x: 0, y: 0, width: 1, height: 1 }}
            canvasW={width}
            canvasH={height}
            dur={dur}
            wave={wave}
          />
        </div>
      ) : null}
    </>
  );
}



function hexToGlow(color, alpha) {
  const raw = String(color || "#ffecb4").replace("#", "");
  if (raw.length !== 6) {
    return `rgba(255, 236, 180, ${alpha})`;
  }
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function CharacterLayer({ character, canvasW, canvasH, frame, dur, wave }) {
  const effects = character.effects || [];
  const color = character.color || "#ffecb4";
  // Person motion is only the set breathe / natural-breathe scale.
  // Float and other translateY keys make the cut-out bob — that is not breathe.
  const personMotion = effects.filter(
    (key) => key === "breathe" || key === "natural-breathe"
  );
  const effectStyle = personMotion.length
    ? computeEffectStyle(personMotion, frame, dur, color)
    : {};
  // Filters on the wrapper recolor skin. Keep motion only; glow stays behind.
  const { filter: _ignoreFilter, opacity: _ignoreOpacity, ...motionStyle } = effectStyle;

  const left = character.bbox.x * canvasW;
  const top = character.bbox.y * canvasH;
  const width = character.bbox.width * canvasW;
  const height = character.bbox.height * canvasH;

  const wantsHalo =
    hasEffect(effects, "glow") ||
    hasEffect(effects, "rim") ||
    hasEffect(effects, "halo");
  const spread = hasEffect(effects, "halo") ? 16 + 14 * wave : 6 + 8 * wave;
  const halo = hexToGlow(color, 0.35 + 0.2 * wave);
  const src = assetSrc(character.src);
  const fillStyle = { width: "100%", height: "100%", objectFit: "fill" };
  const maskStyle = src
    ? {
        WebkitMaskImage: `url(${src})`,
        maskImage: `url(${src})`,
        WebkitMaskSize: "100% 100%",
        maskSize: "100% 100%",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
      }
    : {};

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        pointerEvents: "none",
        transformOrigin: "center center",
        ...motionStyle,
      }}
    >
      {wantsHalo && src ? (
        <Img
          src={src}
          style={{
            ...fillStyle,
            position: "absolute",
            inset: 0,
            filter: `drop-shadow(0px 0px ${spread}px ${halo})`,
            opacity: 0.7,
          }}
        />
      ) : null}
      <Img src={src} style={{ ...fillStyle, position: "relative" }} />
      {src ? (
        <div style={{ position: "absolute", inset: 0, ...maskStyle }}>
          <OverlayFX
            region={{
              ...character,
              x: 0,
              y: 0,
              width: 1,
              height: 1,
            }}
            canvasW={width}
            canvasH={height}
            dur={dur}
            wave={wave}
            faceWash={false}
          />
        </div>
      ) : null}
    </div>
  );
}

function CutoutLayer({ cutout, canvasW, canvasH, frame, dur, wave }) {
  const effects = cutout.effects || [];
  const motion = wantsPixelMotion(effects);
  const color = cutout.color || "#ffecb4";
  const effectStyle = motion ? computeEffectStyle(effects, frame, dur, color) : {};
  const {
    filter: shineFilter,
    opacity: _ignoreOpacity,
    transformOrigin: _ignoreOrigin,
    ...motionStyle
  } = effectStyle;
  const left = cutout.bbox.x * canvasW;
  const top = cutout.bbox.y * canvasH;
  const width = cutout.bbox.width * canvasW;
  const height = cutout.bbox.height * canvasH;
  const origin = cutout.origin === "right"
    ? "88% 50%"
    : cutout.origin === "left"
      ? "12% 50%"
      : "center center";
  const region = {
    ...cutout,
    x: cutout.bbox.x,
    y: cutout.bbox.y,
    width: cutout.bbox.width,
    height: cutout.bbox.height,
    source: cutout.source || "card",
  };

  const hasGlow =
    hasEffect(effects, "glow") ||
    hasEffect(effects, "rim") ||
    hasEffect(effects, "halo") ||
    hasEffect(effects, "gold_pulse") ||
    hasEffect(effects, "neon_pulse");
  const spread = hasEffect(effects, "halo") ? 16 + 16 * wave : 6 + 10 * wave;
  let shadowColor = color;
  if (hasEffect(effects, "gold_pulse")) shadowColor = "#ffd26e";
  if (hasEffect(effects, "neon_pulse")) shadowColor = color || "#00ffc8";

  const glowFilter = hasGlow
    ? `drop-shadow(0px 0px ${spread}px ${shadowColor})`
    : undefined;
  const imgFilter = [glowFilter, shineFilter].filter(Boolean).join(" ") || undefined;

  const cutoutSrc = assetSrc(cutout.src);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        pointerEvents: "none",
        ...motionStyle,
        transformOrigin: origin,
      }}
    >
      <Img
        src={cutoutSrc}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "fill",
          filter: imgFilter,
        }}
      />
      {cutoutSrc ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            WebkitMaskImage: `url(${cutoutSrc})`,
            maskImage: `url(${cutoutSrc})`,
            WebkitMaskSize: "100% 100%",
            maskSize: "100% 100%",
            WebkitMaskRepeat: "no-repeat",
            maskRepeat: "no-repeat",
            pointerEvents: "none",
          }}
        >
          <OverlayFX
            region={{
              ...region,
              x: 0,
              y: 0,
              width: 1,
              height: 1,
            }}
            canvasW={width}
            canvasH={height}
            dur={dur}
            wave={wave}
            faceWash={false}
          />
        </div>
      ) : null}
    </div>
  );
}


function isFrontCutout(item) {
  return Boolean(item && item.front);
}

export const Promo = ({ poster, originalSrc: originalSrcProp, regions, characters, cutouts }) => {
  const { durationInFrames: dur, width, height } = useVideoConfig();
  const frame = useCurrentFrame();
  const wave = useLoopWave();
  const posterSrc = assetSrc(poster);
  const originalSrc = assetSrc(originalSrcProp) || posterSrc;
  const allRegions = Array.isArray(regions) ? regions : [];
  const allChars = Array.isArray(characters) ? characters : [];
  const allCutouts = Array.isArray(cutouts) ? cutouts : [];

  const people = allRegions.filter(isPersonRegion);
  // card / button / title are fully handled by CutoutLayer (effects included).
  // Keep them out of UiLayer so no effect leaks through as a rectangle.
  const ui = allRegions.filter((r) => isUiRegion(r) && !isPlaqueCutout(r));
  const plaqueCutouts = allCutouts.filter(isPlaqueCutout);
  const plaqueBack = plaqueCutouts.filter((item) => !isFrontCutout(item));
  const plaqueFront = plaqueCutouts.filter((item) => isFrontCutout(item));
  const propCutouts = allCutouts.filter((item) => !isPlaqueCutout(item));

  return (
    <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>
      {posterSrc ? (
        <Img src={posterSrc} style={{ width, height, objectFit: "fill" }} />
      ) : null}

      {posterSrc
        ? ui.map((region, idx) => (
            <UiLayer
              key={`ui-${region.key || idx}`}
              region={region}
              posterSrc={posterSrc}
              originalSrc={originalSrc}
              canvasW={width}
              canvasH={height}
              frame={frame}
              dur={dur}
              wave={wave}
            />
          ))
        : null}

      {plaqueBack.map((item) => (
        <CutoutLayer
          key={`cutout-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}

      {allChars.length > 0
        ? allChars.map((char) => (
            <CharacterLayer
              key={`char-${char.index}`}
              character={char}
              canvasW={width}
              canvasH={height}
              frame={frame}
              dur={dur}
              wave={wave}
            />
          ))
        : people.map((region, idx) => (
            <OverlayFX
              key={`person-${region.key || idx}`}
              region={region}
              canvasW={width}
              canvasH={height}
              dur={dur}
              wave={wave}
            />
          ))}

      {plaqueFront.map((item) => (
        <CutoutLayer
          key={`cutout-front-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}

      {propCutouts.map((item) => (
        <CutoutLayer
          key={`prop-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}
    </AbsoluteFill>
  );
};
