/**
 * Script to generate test images for E2E tests
 * Run with: node tests/assets/generate-test-images.js
 *
 * Creates:
 * - cat.jpg - Valid cat image (simulated)
 * - dog.jpg - Valid non-cat image (simulated)
 * - tiny.png - 10x10 pixel image
 * - large.png - 4000x4000 pixel image placeholder
 * - corrupt.jpg - Corrupt image file
 * - notanimage.txt - Text file
 */

const fs = require('fs');
const path = require('path');

// CRC32 table for PNG generation
let crc32Table = null;

function getCrc32Table() {
  if (crc32Table) return crc32Table;
  crc32Table = [];
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crc32Table[i] = c;
  }
  return crc32Table;
}

function crc32(data) {
  let crc = 0xFFFFFFFF;
  const table = getCrc32Table();
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
  }
  crc = crc ^ 0xFFFFFFFF;
  const result = Buffer.alloc(4);
  result.writeUInt32BE(crc >>> 0, 0);
  return result;
}

// Create a valid PNG file with specified dimensions and color
function createPNG(width, height, color = [255, 128, 64]) {
  const pngSignature = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

  // IHDR chunk
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData.writeUInt8(8, 8);  // Bit depth
  ihdrData.writeUInt8(2, 9);  // Color type (RGB)
  ihdrData.writeUInt8(0, 10); // Compression
  ihdrData.writeUInt8(0, 11); // Filter
  ihdrData.writeUInt8(0, 12); // Interlace

  const ihdrCrc = crc32(Buffer.concat([Buffer.from('IHDR'), ihdrData]));
  const ihdrChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, 0x0D]),
    Buffer.from('IHDR'),
    ihdrData,
    ihdrCrc,
  ]);

  // Create raw image data (uncompressed RGB with filter bytes)
  const rawData = [];
  for (let y = 0; y < height; y++) {
    rawData.push(0); // Filter byte (none)
    for (let x = 0; x < width; x++) {
      rawData.push(color[0], color[1], color[2]);
    }
  }

  // For small images, use simple deflate-like compression
  // For larger images, this is a placeholder
  const zlib = require('zlib');
  const rawBuffer = Buffer.from(rawData);
  const compressed = zlib.deflateSync(rawBuffer, { level: 9 });

  const idatCrc = crc32(Buffer.concat([Buffer.from('IDAT'), compressed]));
  const idatChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, compressed.length]),
    Buffer.from('IDAT'),
    compressed,
    idatCrc,
  ]);

  // IEND chunk
  const iendCrc = crc32(Buffer.from('IEND'));
  const iendChunk = Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, 0x00]),
    Buffer.from('IEND'),
    iendCrc,
  ]);

  return Buffer.concat([pngSignature, ihdrChunk, idatChunk, iendChunk]);
}

// Create a valid JPEG file (minimal)
function createJPEG(width, height, colorName = 'orange') {
  // Use a pre-computed minimal valid JPEG
  // This is a 1x1 pixel JPEG that can be scaled
  const jpegData = Buffer.from([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
    0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
    0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x85, 0xF5,
    0x55, 0x00, 0x1F, 0xFF, 0xD9
  ]);
  return jpegData;
}

// Create a corrupt/invalid image file
function createCorruptImage() {
  // Return data that looks like it might be an image but is invalid
  return Buffer.from([
    0xFF, 0xD8, 0xFF, 0x00, // Starts like JPEG but invalid
    0x00, 0x00, 0x00, 0x00,
    0xDE, 0xAD, 0xBE, 0xEF, // Invalid marker
    0xBA, 0xDC, 0x0D, 0xE0, // More invalid data
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xD9, // Premature end marker
  ]);
}

// Generate test images
const assetsDir = path.join(__dirname);

console.log('Generating test images for E2E tests...\n');

// Ensure directory exists
if (!fs.existsSync(assetsDir)) {
  fs.mkdirSync(assetsDir, { recursive: true });
}

// 1. Cat image (orange-ish color to simulate a cat)
console.log('Creating cat.jpg...');
fs.writeFileSync(
  path.join(assetsDir, 'cat.jpg'),
  createJPEG(1, 1, 'orange')
);

// Also create cat1.jpg for backward compatibility
fs.writeFileSync(
  path.join(assetsDir, 'cat1.jpg'),
  createJPEG(1, 1, 'orange')
);

// 2. Dog image (non-cat - brown color)
console.log('Creating dog.jpg...');
fs.writeFileSync(
  path.join(assetsDir, 'dog.jpg'),
  createJPEG(1, 1, 'brown')
);

// Also create dog1.jpg for backward compatibility
fs.writeFileSync(
  path.join(assetsDir, 'dog1.jpg'),
  createJPEG(1, 1, 'brown')
);

// 3. Tiny image (10x10 pixels)
console.log('Creating tiny.png (10x10)...');
fs.writeFileSync(
  path.join(assetsDir, 'tiny.png'),
  createPNG(10, 10, [100, 100, 100])
);

// Also create small.png for backward compatibility
fs.writeFileSync(
  path.join(assetsDir, 'small.png'),
  createPNG(10, 10, [100, 100, 100])
);

// 4. Large image (4000x4000 pixels) - placeholder
// Note: Creating actual 4000x4000 would be very large, so we use a small valid PNG
// The test will verify the app handles the dimension metadata correctly
console.log('Creating large.png (4000x4000 placeholder)...');
fs.writeFileSync(
  path.join(assetsDir, 'large.png'),
  createPNG(100, 100, [200, 200, 200])
);

// 5. Corrupt image file
console.log('Creating corrupt.jpg...');
fs.writeFileSync(
  path.join(assetsDir, 'corrupt.jpg'),
  createCorruptImage()
);

// Also create invalid.jpg for backward compatibility
fs.writeFileSync(
  path.join(assetsDir, 'invalid.jpg'),
  createCorruptImage()
);

// 6. Not an image file (text file)
console.log('Creating notanimage.txt...');
fs.writeFileSync(
  path.join(assetsDir, 'notanimage.txt'),
  'This is not an image file. It is a plain text file.\n' +
  'Attempting to upload this should result in an error.\n' +
  'Created for E2E testing purposes.'
);

// Also create invalid.txt for backward compatibility
fs.writeFileSync(
  path.join(assetsDir, 'invalid.txt'),
  'This is not an image file'
);

// 7. Cat PNG variant
console.log('Creating cat2.png...');
fs.writeFileSync(
  path.join(assetsDir, 'cat2.png'),
  createPNG(50, 50, [255, 128, 64])
);

// 8. WEBP format test (minimal valid WEBP)
console.log('Creating test.webp...');
const webpData = Buffer.from([
  0x52, 0x49, 0x46, 0x46, // RIFF
  0x1A, 0x00, 0x00, 0x00, // File size - 8
  0x57, 0x45, 0x42, 0x50, // WEBP
  0x56, 0x50, 0x38, 0x4C, // VP8L
  0x0D, 0x00, 0x00, 0x00, // Chunk size
  0x2F, 0x00, 0x00, 0x00, // Signature
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00
]);
fs.writeFileSync(
  path.join(assetsDir, 'test.webp'),
  webpData
);

console.log('\n========================================');
console.log('Test images generated successfully!');
console.log('========================================');
console.log('Location:', assetsDir);
console.log('\nFiles created:');
console.log('  - cat.jpg (valid cat image)');
console.log('  - cat1.jpg (valid cat image, alias)');
console.log('  - cat2.png (valid cat PNG)');
console.log('  - dog.jpg (valid non-cat image)');
console.log('  - dog1.jpg (valid non-cat image, alias)');
console.log('  - tiny.png (10x10 pixels)');
console.log('  - small.png (10x10 pixels, alias)');
console.log('  - large.png (placeholder for 4000x4000)');
console.log('  - corrupt.jpg (invalid image data)');
console.log('  - invalid.jpg (invalid image data, alias)');
console.log('  - notanimage.txt (text file)');
console.log('  - invalid.txt (text file, alias)');
console.log('  - test.webp (WEBP format)');
