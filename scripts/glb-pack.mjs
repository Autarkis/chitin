const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

function padded(length) {
  return (length + 3) & ~3;
}

export function packGlb(document, binary, declaredBinaryLength) {
  const source = binary instanceof ArrayBuffer
    ? new Uint8Array(binary)
    : new Uint8Array(binary.buffer, binary.byteOffset, binary.byteLength);
  const contentLength = declaredBinaryLength ?? source.byteLength;
  if (contentLength > source.byteLength) {
    throw new RangeError("declared GLB binary length exceeds the supplied data");
  }

  const jsonBytes = new TextEncoder().encode(JSON.stringify(document));
  const jsonLength = padded(jsonBytes.byteLength);
  const binaryLength = padded(contentLength);
  const total = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new ArrayBuffer(total);
  const view = new DataView(output);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, JSON_CHUNK, true);
  const jsonOutput = new Uint8Array(output, 20, jsonLength);
  jsonOutput.fill(0x20);
  jsonOutput.set(jsonBytes);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binaryLength, true);
  view.setUint32(binaryHeader + 4, BIN_CHUNK, true);
  new Uint8Array(output, binaryHeader + 8, contentLength).set(
    source.subarray(0, contentLength),
  );
  return output;
}
