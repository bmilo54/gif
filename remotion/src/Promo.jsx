import React from "react";
import { Lottie } from "@remotion/lottie";
import {
  AbsoluteFill,
  Easing,
  Img,
  OffthreadVideo,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import glowData from "./lottie/glow.json";
import shineData from "./lottie/shine.json";
import sparkleData from "./lottie/sparkle.json";

function assetSrc(src) {
  if (!src) {
    return null;
  }
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

const UI_SOURCES = new Set(["card", "button", "title", "ocr"]);
const LOTTIE_PRESETS = {
  shine: shineData,
  sparkle: sparkleData,
  glow: glowData,
};

function isUiRegion(region) {
  return UI_SOURCES.has((region.source || "").toLowerCase());
}

function hasEffect(effects, name) {
  return Array.isArray(effects) && effects.includes(name);
}

function boxPixels(region, canvasWidth, canvasHeight) {
  const left = region.x * canvasWidth;
  const top = region.y * canvasHeight;
  const width = region.width * canvasWidth;
  const height = region.height * canvasHeight;
  const radius =
    (region.source || "").toLowerCase() === "button"
      ? Math.max(4, height / 2)
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

function RegionSlice({ region, poster, canvasWidth, canvasHeight, scale, translateY, filter }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasWidth, canvasHeight);
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
      <Img
        src={poster}
        style={{
          position: "absolute",
          left: -left,
          top: -top,
          width: canvasWidth,
          height: canvasHeight,
          objectFit: "fill",
          transform: `translateY(${translateY}px) scale(${scale})`,
          transformOrigin: `${left + width / 2}px ${top + height / 2}px`,
          filter,
        }}
      />
    </div>
  );
}

function lottieNames(effects) {
  const names = [];
  if (hasEffect(effects, "shine")) {
    names.push("shine");
  }
  if (hasEffect(effects, "sparkle")) {
    names.push("sparkle");
  }
  if (
    hasEffect(effects, "glow") ||
    hasEffect(effects, "gold_pulse") ||
    hasEffect(effects, "rim") ||
    hasEffect(effects, "fade") ||
    hasEffect(effects, "flicker")
  ) {
    names.push("glow");
  }
  return names;
}

function LottieBox({ region, canvasWidth, canvasHeight, animationData, durationInFrames }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasWidth, canvasHeight);
  const ip = Number(animationData.ip) || 0;
  const op = Number(animationData.op) || 60;
  const playbackRate = (op - ip) / Math.max(durationInFrames, 1);
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
      <Lottie
        animationData={animationData}
        playbackRate={playbackRate}
        loop
        style={{ width: "100%", height: "100%" }}
        preserveAspectRatio="none"
      />
    </div>
  );
}

function PersonLayer({ src, region, canvasWidth, canvasHeight }) {
  if (!src || !region || !(region.width > 0) || !(region.height > 0)) {
    return null;
  }
  const { left, top, width, height } = boxPixels(region, canvasWidth, canvasHeight);
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      <OffthreadVideo
        src={src}
        muted
        volume={0}
        loop
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
    </div>
  );
}

export const Promo = ({ poster, person, personRegion, effects, regions }) => {
  const { durationInFrames, width, height } = useVideoConfig();
  const wave = useLoopWave();
  const list = Array.isArray(effects) ? effects : [];
  const boxes = Array.isArray(regions) ? regions : [];
  const overlayBoxes = boxes.filter(isUiRegion);
  const posterSrc = assetSrc(poster);
  const personSrc = assetSrc(person);
  const overlays = lottieNames(list);
  const progress = useCurrentFrame() / Math.max(durationInFrames, 1);

  const needsSlice =
    hasEffect(list, "zoom") || hasEffect(list, "float") || hasEffect(list, "breathe");

  let scale = 1;
  if (hasEffect(list, "zoom")) {
    scale += interpolate(wave, [0, 1], [0, 0.08], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }
  if (hasEffect(list, "breathe")) {
    scale += interpolate(wave, [0, 1], [0, 0.03], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }

  const translateY = hasEffect(list, "float")
    ? interpolate(progress, [0, 0.5, 1], [0, -6, 0], {
        easing: Easing.inOut(Easing.sin),
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 0;

  return (
    <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>
      {posterSrc ? <Img src={posterSrc} style={{ width, height, objectFit: "fill" }} /> : null}

      <PersonLayer
        src={personSrc}
        region={personRegion}
        canvasWidth={width}
        canvasHeight={height}
      />

      {needsSlice && posterSrc
        ? overlayBoxes.map((region, index) => (
            <RegionSlice
              key={`slice-${index}`}
              region={region}
              poster={posterSrc}
              canvasWidth={width}
              canvasHeight={height}
              scale={scale}
              translateY={translateY}
              filter="none"
            />
          ))
        : null}

      {overlays.flatMap((name) =>
        overlayBoxes.map((region, index) => (
          <LottieBox
            key={`lottie-${name}-${index}`}
            region={region}
            canvasWidth={width}
            canvasHeight={height}
            animationData={LOTTIE_PRESETS[name]}
            durationInFrames={durationInFrames}
          />
        )),
      )}
    </AbsoluteFill>
  );
};
