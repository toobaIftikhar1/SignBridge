export const NUM_POINTS = 21; // MediaPipe gives 21 points for each hand
export const PER_HAND = NUM_POINTS * 3 + 2; // 21 points x 3 (x, y, z) = 63, plus wrist x and y = 65
export const PER_FRAME = PER_HAND * 2; // 2 hands x 65 = 130 numbers for one frame
export const SEQ_LEN = 30; // a moving sign is always cut down to 30 frames

/**
 * Takes the raw result from MediaPipe and makes a simple list of hands.
 * Each hand is a list of 21 points, and each point is [x, y, z].
 */
export function resultToFrame(mediaPipeResult) {
  // If no hands were found, return an empty list
  if (!mediaPipeResult?.landmarks) return [];
  return mediaPipeResult.landmarks.map((hand) =>
    hand.map((point) => [roundTo4Decimals(point.x), roundTo4Decimals(point.y), roundTo4Decimals(point.z)])
  );
}

// Keeps only 4 digits after the decimal point (0.123456 becomes 0.1235)
function roundTo4Decimals(value) {
  return Math.round(value * 10000) / 10000;
}

/** One hand (21 points) -> 65 numbers. */
export function handFeatures(hand) {
  // Point number 0 is the wrist. We remember its x, y and z.
  const [wristX, wristY, wristZ] = hand[0];

  // Step 1: subtract the wrist from every point, so the wrist becomes (0, 0, 0).
  // Now the numbers do not depend on where the hand is in the camera.
  const pointsFromWrist = hand.map(([x, y, z]) => [x - wristX, y - wristY, z - wristZ]);

  // Find the point that is the farthest from the wrist (the biggest distance).
  let farthestDistance = 0;
  for (const [x, y, z] of pointsFromWrist) {
    farthestDistance = Math.max(farthestDistance, Math.sqrt(x * x + y * y + z * z));
  }

  // If the distance is almost zero, use 1 so we never divide by zero
  if (farthestDistance < 1e-6) farthestDistance = 1;

  const numbers = [];

  // Step 2: divide every number by the farthest distance.
  // Now a big hand and a small hand give the same numbers. This gives 63 numbers.
  for (const [x, y, z] of pointsFromWrist) {
    numbers.push(x / farthestDistance, y / farthestDistance, z / farthestDistance);
  }

  // Step 3: add the wrist position (image centre becomes 0, 0). This gives 2 more numbers.
  numbers.push(wristX - 0.5, wristY - 0.5);

  return numbers; // 63 + 2 = 65 numbers
}

/** One frame (0, 1 or 2 hands) -> 130 numbers. */
export function frameFeatures(frame) {
  // Sort hands by wrist x, so the hand on the left side of the image comes first.
  // Keep only 2 hands at most.
  const sortedHands = [...(frame || [])].sort((a, b) => a[0][0] - b[0][0]).slice(0, 2);

  // 65 zeros, used when a hand is missing
  const emptyHand = new Array(PER_HAND).fill(0);

  // First slot = left hand, second slot = the other hand.
  // If only one hand is visible, it always goes in the first slot.
  const firstHandNumbers = sortedHands[0] ? handFeatures(sortedHands[0]) : emptyHand;
  const secondHandNumbers = sortedHands[1] ? handFeatures(sortedHands[1]) : emptyHand;

  return [...firstHandNumbers, ...secondHandNumbers]; // 65 + 65 = 130
}

/** Picks 30 frames spread evenly from a recording: index i = floor(i*(n-1)/29 + 0.5). */
export function pickFrames(frames, framesWanted = SEQ_LEN) {
  const totalFrames = frames.length;
  if (totalFrames === 0) return [];
  const pickedFrames = [];
  for (let i = 0; i < framesWanted; i++) {
    pickedFrames.push(frames[Math.floor((i * (totalFrames - 1)) / (framesWanted - 1) + 0.5)]);
  }
  return pickedFrames;
}

/** Moving sign: list of frames -> 30 x 130 numbers. */
export function dynamicFeatures(frames) {
  return pickFrames(frames).map(frameFeatures);
}

/** Still sign: one frame -> 130 numbers. */
export function staticFeatures(frame) {
  return frameFeatures(frame);
}

// True if the frame has at least one hand in it
export function hasHands(frame) {
  return Array.isArray(frame) && frame.length > 0;
}