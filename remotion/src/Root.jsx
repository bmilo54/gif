import React from "react";
import { Composition } from "remotion";
import { Promo } from "./Promo";

export const RemotionRoot = () => {
  return (
    <Composition
      id="Promo"
      component={Promo}
      durationInFrames={32}
      fps={20}
      width={720}
      height={720}
      defaultProps={{
        poster: "",
        person: "",
        personRegion: {},
        effects: ["shine"],
        regions: [],
        durationInFrames: 32,
        fps: 20,
        width: 720,
        height: 720,
      }}
      calculateMetadata={({ props }) => ({
        durationInFrames: Number(props.durationInFrames) || 32,
        fps: Number(props.fps) || 20,
        width: Number(props.width) || 720,
        height: Number(props.height) || 720,
      })}
    />
  );
};
