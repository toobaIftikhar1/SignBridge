// This file reads raw landmark data, converts it into features, and saves the result.
import { readFileSync, writeFileSync } from 'node:fs';   // Read and write files
import { staticFeatures, dynamicFeatures } from '../web/src/lib/landmarks.js';   // Feature processing functions

const [input, output] = process.argv.slice(2);    // Get input and output file names
const cases = JSON.parse(readFileSync(input, 'utf8'));    // Read and convert input JSON
const result = cases.map((c) => ({    // Process each case
  static: staticFeatures(c.frames[0]),
  dynamic: dynamicFeatures(c.frames),
}));
writeFileSync(output, JSON.stringify(result));    // Save processed data as JSON
