#include <stdio.h>
#include <string.h>

void IMG_erode_bin_c
(
    const unsigned char * in_data,
    unsigned char       * out_data,
    const char          * mask,
    int cols
);

/* ======================================================================== */
/*  Constant dataset.                                                       */
/* ======================================================================== */
#define N    (2048)
#define COLS (N/8)

/* ======================================================================== */
/*  Initialize arrays with random test data.                                */
/* ======================================================================== */
const unsigned char  in_data[N] =
{
     0xBA,  0x2E,  0xEA,  0x9E,  0x3A,  0xED,  0xE7,  0xB0,
     0xEA,  0x09,  0xB8,  0xC4,  0x4C,  0x13,  0xEA,  0x76,
     0x7A,  0x6F,  0x1A,  0x71,  0x67,  0xB2,  0xA4,  0x19,
     0x33,  0x77,  0x75,  0xDE,  0xBC,  0x5A,  0x0D,  0xD0,
     0xA8,  0xE4,  0x52,  0x35,  0xB2,  0x2C,  0x02,  0xF8,
     0x30,  0xA9,  0xD4,  0x4E,  0xC1,  0x5F,  0xA4,  0xB1,
     0x5E,  0xE7,  0x8F,  0xD9,  0x78,  0xF8,  0x91,  0xEF,
     0xB3,  0x6A,  0x60,  0x03,  0xDB,  0xC6,  0xAA,  0xB8,
     0x88,  0xF2,  0x7F,  0x46,  0x15,  0x80,  0x02,  0x67,
     0xBC,  0x8D,  0x94,  0xC7,  0xAD,  0x96,  0x59,  0x8E,
     0xED,  0x7F,  0x94,  0xA3,  0x54,  0xF9,  0xDD,  0x47,
     0xAF,  0x02,  0x7A,  0x81,  0xA9,  0x6B,  0x5B,  0xFF,
     0x82,  0x10,  0x55,  0x2A,  0x3E,  0xA8,  0xE5,  0x4C,
     0xEA,  0x1F,  0x2A,  0x06,  0x88,  0xB0,  0xF9,  0xC2,
     0x8B,  0x75,  0xFD,  0x35,  0x68,  0x06,  0x9F,  0xC5,
     0xCB,  0x29,  0xF4,  0x5F,  0x54,  0xA2,  0x77,  0xE4,
     0x12,  0x82,  0xF6,  0x79,  0x4D,  0x94,  0xD2,  0xAF,
     0x61,  0x9F,  0x4B,  0x83,  0x39,  0x07,  0xE9,  0xE5,
     0xB7,  0xCA,  0xC8,  0x97,  0x5F,  0x85,  0xE9,  0x6C,
     0x5D,  0xD2,  0xBC,  0xDC,  0x56,  0xCC,  0x3B,  0xE8,
     0xF9,  0x16,  0x7E,  0x72,  0xCB,  0x75,  0x89,  0x91,
     0x0F,  0xC0,  0x8E,  0x16,  0x47,  0x0C,  0x38,  0x94,
     0x5D,  0xAD,  0xAD,  0xF2,  0x33,  0xE2,  0xD9,  0x8F,
     0x61,  0x53,  0xC7,  0x17,  0x26,  0x25,  0xBA,  0x7F,
     0x1F,  0x9C,  0x8B,  0x2B,  0xC3,  0x24,  0x32,  0x2D,
     0x3C,  0x60,  0xC2,  0x1A,  0x2C,  0xDC,  0x64,  0x44,
     0x5E,  0x9E,  0xCB,  0x08,  0x75,  0x04,  0x4B,  0x12,
     0xBE,  0x1B,  0x90,  0x4B,  0x48,  0xC1,  0x88,  0xA3,
     0xD9,  0x50,  0x28,  0xF8,  0x5E,  0x49,  0x10,  0xDB,
     0x23,  0xD8,  0xBE,  0x69,  0x40,  0x43,  0x16,  0x41,
     0xC0,  0xC6,  0xE1,  0x35,  0x59,  0xC9,  0xB2,  0x7F,
     0x59,  0x40,  0x54,  0x35,  0x38,  0x35,  0xB5,  0xE7,
     0x25,  0xC4,  0x7C,  0xA2,  0x78,  0xD4,  0xA4,  0xF5,
     0x96,  0x68,  0x1A,  0xC7,  0xC1,  0xA7,  0x70,  0x65,
     0x0F,  0x11,  0xA0,  0x6D,  0xF8,  0x14,  0x97,  0x66,
     0x6E,  0x62,  0x77,  0x13,  0xB2,  0x60,  0x86,  0x6B,
     0x4C,  0xC3,  0xDA,  0xD2,  0xB7,  0x34,  0xA5,  0x57,
     0x38,  0x39,  0x6E,  0x1F,  0x80,  0x55,  0x56,  0xFC,
     0x68,  0x73,  0x94,  0xBF,  0x56,  0xDB,  0x38,  0x98,
     0xC4,  0x70,  0xA7,  0xCC,  0xD8,  0xE7,  0x71,  0x53,
     0x79,  0xA6,  0xA3,  0x3E,  0x98,  0xE2,  0x68,  0xEB,
     0xBA,  0x79,  0x73,  0x2A,  0xA0,  0xC7,  0x43,  0x5B,
     0xAE,  0xF6,  0x1E,  0xD0,  0x9C,  0xC3,  0xD3,  0xA8,
     0x19,  0xBA,  0x33,  0xB4,  0xB4,  0xD3,  0xEA,  0x1B,
     0xB3,  0xAD,  0x7A,  0x24,  0x00,  0xAC,  0x2B,  0xCB,
     0xAC,  0x0F,  0xAB,  0xF7,  0xF8,  0x40,  0x20,  0xA5,
     0x75,  0xF5,  0x06,  0xE4,  0x1A,  0xBF,  0x0D,  0x45,
     0x73,  0xE8,  0x10,  0x73,  0x99,  0xDD,  0x62,  0x71,
     0xCD,  0xC4,  0x52,  0x73,  0xC6,  0x50,  0xC3,  0x6C,
     0x58,  0x6B,  0xED,  0xCD,  0x94,  0x5E,  0x9E,  0x44,
     0x15,  0x17,  0x1E,  0xDB,  0x7F,  0x44,  0x0D,  0xE3,
     0xA8,  0x41,  0x1C,  0x8B,  0xE0,  0xDE,  0x21,  0xED,
     0xBE,  0x4A,  0xCB,  0x45,  0xBB,  0xDA,  0x2D,  0x62,
     0x29,  0xEC,  0x79,  0xF9,  0xCE,  0x14,  0x99,  0x91,
     0x62,  0x7C,  0x80,  0xB9,  0xCC,  0x78,  0xA8,  0x30,
     0xC5,  0xD1,  0x12,  0x00,  0x7F,  0xED,  0x4F,  0x46,
     0xE6,  0x54,  0x6C,  0x0E,  0xAD,  0x4F,  0x8E,  0x31,
     0x24,  0x5E,  0xEC,  0xF1,  0x84,  0x83,  0xFD,  0x31,
     0xA4,  0xAB,  0xDF,  0x2B,  0x8A,  0xF1,  0x43,  0xE3,
     0xB4,  0xE4,  0xAF,  0xA3,  0x01,  0xA0,  0xD6,  0x75,
     0xC3,  0xDA,  0x28,  0x19,  0xEB,  0x21,  0x09,  0xAE,
     0xE7,  0x18,  0xBC,  0x60,  0xF1,  0x49,  0xB3,  0x9B,
     0x26,  0xB4,  0x66,  0x89,  0xBE,  0x81,  0xD1,  0x54,
     0xB8,  0x5A,  0x96,  0x7B,  0x5C,  0xFF,  0x78,  0x47,
     0x92,  0x70,  0xC1,  0xB7,  0xC4,  0xE9,  0x09,  0x75,
     0xB9,  0x27,  0x9A,  0x93,  0x9F,  0xB2,  0x11,  0x59,
     0x05,  0x0C,  0xB8,  0xE2,  0x0D,  0x81,  0x7A,  0x69,
     0x33,  0x59,  0x5F,  0xD2,  0x1E,  0x96,  0xB7,  0xE0,
     0x59,  0xE8,  0x7A,  0xBB,  0x81,  0xEE,  0x62,  0xEB,
     0x28,  0x20,  0x54,  0xC0,  0xF2,  0x51,  0x4C,  0x6F,
     0xA0,  0xA3,  0x7A,  0x80,  0xB0,  0x2D,  0x62,  0x54,
     0x38,  0xD3,  0x60,  0x5F,  0x7B,  0x36,  0xFA,  0xB6,
     0xE5,  0x79,  0xA5,  0xB0,  0x61,  0xEE,  0xE5,  0x59,
     0xB4,  0x00,  0xA6,  0xB8,  0xD9,  0x85,  0x76,  0x80,
     0x38,  0xB1,  0xFC,  0x76,  0x96,  0x85,  0x1E,  0xA0,
     0x6C,  0x64,  0xAB,  0x14,  0xBB,  0xDF,  0x7C,  0x70,
     0x2E,  0xD5,  0x79,  0x55,  0x0A,  0xBD,  0xCB,  0x8B,
     0xFF,  0xA8,  0x82,  0x72,  0x13,  0x69,  0x7B,  0x18,
     0x40,  0xBE,  0x6C,  0xA3,  0x4B,  0x3A,  0x44,  0x11,
     0xB2,  0x0F,  0xBF,  0x0D,  0x7F,  0x50,  0x7A,  0x45,
     0xAE,  0xBF,  0xCD,  0xD6,  0x02,  0x1E,  0x93,  0x8D,
     0x19,  0x77,  0x4D,  0xF9,  0x98,  0xA4,  0xA8,  0xAB,
     0xD6,  0x63,  0x29,  0xC4,  0x15,  0x23,  0xA4,  0xD3,
     0x24,  0x50,  0x70,  0x20,  0x64,  0xC0,  0x2E,  0x3C,
     0x20,  0x99,  0x1E,  0x48,  0x9A,  0x6D,  0xB8,  0xF3,
     0x4A,  0x76,  0x8D,  0x3D,  0x9B,  0xAF,  0x2F,  0xCC,
     0xEC,  0x41,  0xA1,  0xFC,  0xD9,  0xB6,  0xAC,  0x7D,
     0xE0,  0x05,  0x80,  0x8A,  0x88,  0x48,  0x07,  0xED,
     0x2F,  0x68,  0x6A,  0xE6,  0xDC,  0x95,  0xD4,  0x31,
     0xF2,  0x9F,  0x95,  0x44,  0xAB,  0xF8,  0x3B,  0x08,
     0x9D,  0xB5,  0x4E,  0x61,  0x31,  0x13,  0x63,  0x62,
     0x12,  0xB5,  0x76,  0x44,  0x69,  0x67,  0xAD,  0x27,
     0x94,  0xD1,  0x65,  0x8D,  0xF6,  0xC4,  0xA8,  0x87,
     0xF4,  0xC3,  0x9D,  0x3D,  0x6F,  0x75,  0xFF,  0xFD,
     0x2D,  0xC4,  0x3E,  0x0E,  0x47,  0x11,  0x9B,  0xB3,
     0xDD,  0xAB,  0x1E,  0x74,  0x9C,  0x3E,  0x42,  0x0E,
     0x19,  0x96,  0x80,  0xFB,  0xA2,  0x76,  0x84,  0xFC,
     0x2D,  0x7D,  0xD4,  0xF0,  0x81,  0x1E,  0x4F,  0x1D,
     0x38,  0xBC,  0xB8,  0x2B,  0xEA,  0xD7,  0xB5,  0xF7,
     0x89,  0x6F,  0x49,  0x7F,  0xEA,  0xEB,  0x73,  0x26,
     0x12,  0x02,  0x42,  0x50,  0x07,  0x3C,  0x99,  0xEB,
     0x78,  0xCC,  0xE5,  0x11,  0xE4,  0xD2,  0x41,  0xA2,
     0x91,  0x15,  0x72,  0x86,  0x5E,  0x7F,  0x9C,  0x4B,
     0x76,  0x16,  0x39,  0x3E,  0x43,  0x95,  0xC3,  0xCF,
     0xBE,  0xD3,  0x81,  0x4E,  0x99,  0xD6,  0xAC,  0xA0,
     0xC0,  0xB8,  0x71,  0xFB,  0x99,  0x1E,  0x4A,  0x35,
     0x47,  0x04,  0xDD,  0x1B,  0x70,  0x3B,  0x55,  0xFC,
     0x6F,  0xE7,  0x48,  0xD6,  0x15,  0x96,  0xA0,  0xB5,
     0x17,  0x21,  0xDE,  0xD9,  0x7D,  0x0F,  0xB3,  0x8A,
     0x9F,  0x9E,  0x00,  0x3F,  0xD1,  0x78,  0x66,  0xC5,
     0x5A,  0x4D,  0x03,  0x2C,  0x3F,  0x8F,  0xC2,  0xEE,
     0xB6,  0xDE,  0x42,  0xAA,  0x74,  0x5A,  0xC1,  0xB3,
     0x8F,  0xB3,  0x2A,  0x3C,  0xDC,  0xE8,  0xCE,  0x74,
     0x07,  0x89,  0xE5,  0x9D,  0x38,  0x59,  0xC8,  0x0F,
     0xA4,  0xCA,  0x82,  0x53,  0x45,  0x0A,  0xFD,  0x50,
     0x6A,  0x8A,  0xF6,  0xF4,  0xAF,  0x60,  0x3D,  0x9B,
     0x36,  0x5A,  0x10,  0xA9,  0xC3,  0x85,  0x34,  0x5C,
     0x7F,  0x2A,  0x96,  0xDF,  0xE5,  0xC6,  0x87,  0x55,
     0x61,  0x2B,  0x12,  0x14,  0x20,  0x0B,  0xE5,  0x2B,
     0xB7,  0xB8,  0x75,  0x7E,  0xB3,  0xE6,  0xF7,  0x80,
     0xD2,  0xBC,  0x9C,  0x82,  0xEF,  0x69,  0x74,  0x52,
     0x63,  0xE8,  0xFE,  0x0C,  0x5E,  0xBF,  0xD7,  0xC9,
     0xF2,  0x73,  0x29,  0x74,  0x94,  0x0F,  0x44,  0x3A,
     0x50,  0x8B,  0x8F,  0xFC,  0xD6,  0xBB,  0xC2,  0xE2,
     0x66,  0x1A,  0x17,  0xDB,  0x27,  0x79,  0xA5,  0xEA,
     0xBD,  0x8D,  0x3D,  0x87,  0x3D,  0x1B,  0x43,  0x2A,
     0x41,  0xA7,  0x2D,  0x0F,  0x4B,  0x0C,  0x23,  0x92,
     0xB8,  0x57,  0x02,  0x83,  0x78,  0xC0,  0xCD,  0x55,
     0x9A,  0x90,  0xB3,  0xC2,  0x86,  0x47,  0xFC,  0xC2,
     0x1E,  0xFF,  0x63,  0x86,  0xF8,  0x44,  0x6C,  0x0E,
     0x5F,  0x3F,  0xBC,  0x2D,  0xD1,  0x4C,  0xBD,  0xA2,
     0xD8,  0xE5,  0xDA,  0xA7,  0x05,  0x8E,  0xDB,  0x16,
     0xA2,  0x51,  0x95,  0xC8,  0x49,  0x28,  0x40,  0x93,
     0x0F,  0xA6,  0x9C,  0x70,  0x36,  0x69,  0xD2,  0x5D,
     0xBF,  0xC9,  0x92,  0x5D,  0x53,  0x5C,  0x9B,  0x00,
     0x65,  0x8B,  0x89,  0xB9,  0xD3,  0xC3,  0xB8,  0x0C,
     0xFF,  0x54,  0xA6,  0x44,  0x96,  0x92,  0xAB,  0x5B,
     0x87,  0xD6,  0x54,  0x99,  0x56,  0x8E,  0x18,  0x83,
     0xAE,  0x46,  0x74,  0x0B,  0x92,  0x7A,  0x54,  0x85,
     0x8B,  0x97,  0x38,  0x9E,  0x59,  0x7E,  0x74,  0xC4,
     0xBE,  0x00,  0xCA,  0xE5,  0xC5,  0x3F,  0xA6,  0x12,
     0xF7,  0xE3,  0xCD,  0xE2,  0x81,  0x0A,  0x5A,  0xCD,
     0x6F,  0xC5,  0xBF,  0x9D,  0x9A,  0x10,  0x6E,  0x3C,
     0x6E,  0xA0,  0xC4,  0xD4,  0x4E,  0x7D,  0xE7,  0xD4,
     0x97,  0x52,  0xDD,  0xF4,  0x9E,  0xA2,  0xF6,  0x6E,
     0x56,  0x43,  0xAC,  0xB2,  0xE1,  0xEE,  0xFE,  0x76,
     0x82,  0xAE,  0x04,  0x8D,  0xAF,  0xBD,  0xEE,  0xA7,
     0xF2,  0x23,  0xFA,  0x0D,  0x15,  0x6D,  0xF7,  0x36,
     0x81,  0xF9,  0x0D,  0xE7,  0x1F,  0x1F,  0xC6,  0x42,
     0xC9,  0x4F,  0xFD,  0xDC,  0x78,  0xD2,  0x8A,  0xEF,
     0xA2,  0x4F,  0x3A,  0x02,  0x04,  0x5E,  0xFD,  0xCA,
     0x53,  0x01,  0x7E,  0xF3,  0x15,  0xD8,  0x25,  0x10,
     0x2F,  0xE9,  0x0A,  0xA7,  0x33,  0x89,  0x15,  0xE4,
     0x4A,  0x1B,  0x26,  0xDE,  0x0C,  0x8F,  0x0A,  0xC8,
     0xE3,  0x1D,  0x06,  0x73,  0x9B,  0xA7,  0x45,  0x8F,
     0x09,  0x51,  0xF2,  0x84,  0x65,  0x44,  0xDA,  0x6F,
     0xFB,  0xEC,  0x86,  0x22,  0xF2,  0x70,  0x11,  0x27,
     0xE7,  0xE3,  0x54,  0x00,  0xB4,  0x61,  0x6B,  0xDC,
     0x8A,  0x31,  0x99,  0xBE,  0xF9,  0xC0,  0x64,  0xFC,
     0x77,  0x10,  0xC4,  0x9A,  0x59,  0xF8,  0x02,  0x74,
     0xC1,  0xAB,  0x8A,  0x99,  0xD0,  0xB5,  0xCE,  0x69,
     0x09,  0x9D,  0xDC,  0xDA,  0x87,  0xED,  0x68,  0x1D,
     0x08,  0x78,  0xA2,  0x4F,  0x02,  0x6F,  0x09,  0xB4,
     0xEC,  0x28,  0x43,  0x0B,  0x67,  0xED,  0x61,  0x6B,
     0x1F,  0xAC,  0x29,  0x2E,  0x59,  0xE6,  0xE7,  0x2E,
     0x45,  0x14,  0x10,  0xB7,  0xDE,  0xB8,  0x77,  0x9F,
     0xA5,  0x25,  0xD1,  0xB9,  0xA0,  0x4A,  0x3C,  0x4E,
     0x6D,  0x4C,  0x57,  0xED,  0xF1,  0x61,  0x37,  0x60,
     0xAD,  0xCC,  0xF8,  0x33,  0xDC,  0xE3,  0x83,  0xCE,
     0x42,  0x28,  0x0B,  0x58,  0xBB,  0x83,  0x39,  0x48,
     0x5C,  0xD4,  0x78,  0x5A,  0xA4,  0x6D,  0x3D,  0xFA,
     0xCC,  0x34,  0x78,  0x7D,  0x67,  0x9D,  0xA3,  0x98,
     0x9A,  0xB5,  0xF0,  0xC1,  0x9E,  0x50,  0x41,  0xB6,
     0x24,  0xA3,  0xED,  0x9E,  0xC6,  0x0D,  0xF3,  0xBF,
     0x60,  0xEC,  0xBB,  0x7D,  0x5B,  0x45,  0x98,  0xF8,
     0x8C,  0xAC,  0x5D,  0x24,  0x44,  0x5A,  0x4D,  0x83,
     0x35,  0xC5,  0x0C,  0x06,  0xFE,  0x48,  0x84,  0xB2,
     0x0C,  0x24,  0x15,  0x8B,  0x73,  0xC5,  0x8A,  0xFA,
     0xCC,  0xB2,  0xD0,  0x7F,  0x72,  0xD5,  0xD2,  0x61,
     0x19,  0x01,  0xD3,  0x21,  0x6D,  0x56,  0xD4,  0x99,
     0xEC,  0xF6,  0x55,  0xD9,  0x25,  0xE2,  0xA7,  0xD3,
     0x19,  0xAB,  0xC6,  0x30,  0xB5,  0xDC,  0x83,  0x0F,
     0x07,  0xA2,  0xE2,  0x6F,  0x82,  0x10,  0x43,  0xC2,
     0xBE,  0x41,  0x57,  0x4D,  0x68,  0x8F,  0x84,  0xB1,
     0x2E,  0x3B,  0xB1,  0x11,  0xA3,  0xFA,  0x98,  0x60,
     0x86,  0x34,  0xC5,  0xEF,  0xDA,  0x4D,  0x8B,  0xAA,
     0x4C,  0x72,  0xFC,  0xB4,  0xEC,  0xB4,  0x8D,  0x00,
     0xF8,  0xDB,  0xDE,  0x85,  0xF7,  0xA8,  0xCB,  0x3E,
     0xC1,  0xD6,  0x0F,  0x18,  0x97,  0xC2,  0x55,  0x56,
     0x47,  0xF2,  0x73,  0xAB,  0x02,  0x3C,  0xFF,  0xBB,
     0xE7,  0x5B,  0x8E,  0x36,  0x61,  0x33,  0x3F,  0x1A,
     0x8D,  0x13,  0x50,  0x3A,  0x89,  0xF6,  0xF9,  0x05,
     0x00,  0x28,  0x65,  0x41,  0xCC,  0x9F,  0xCF,  0xFD,
     0xBA,  0xA4,  0xCD,  0x51,  0xAC,  0x52,  0x01,  0xCC,
     0x9C,  0x0E,  0x09,  0x82,  0xB9,  0x26,  0x1C,  0x33,
     0xF0,  0xD0,  0x40,  0x95,  0x08,  0xB7,  0xC3,  0xC3,
     0x88,  0x7C,  0x07,  0x64,  0x52,  0xCF,  0xB2,  0x0F,
     0xE4,  0x79,  0xF1,  0x40,  0xE8,  0x71,  0x8A,  0x94,
     0xB9,  0xF3,  0x0A,  0x75,  0x74,  0xC2,  0x81,  0x6D,
     0x78,  0x43,  0x50,  0xC6,  0x8C,  0xF1,  0x0B,  0x3F,
     0xD7,  0x43,  0xCD,  0x32,  0x1B,  0x54,  0x91,  0xE8,
     0xB9,  0x1B,  0x8C,  0x1B,  0xA6,  0x5A,  0x04,  0x32,
     0x57,  0x0E,  0xA3,  0xE0,  0x88,  0xDB,  0x7B,  0x19,
     0xDC,  0xE9,  0xD0,  0xF3,  0x63,  0x7C,  0x69,  0x51,
     0x4B,  0x57,  0x95,  0xCE,  0x10,  0xAF,  0xF9,  0x54,
     0xF5,  0x47,  0xC2,  0x74,  0x99,  0xBF,  0xEA,  0x9D,
     0x4A,  0x2C,  0x31,  0xB8,  0xE1,  0xF2,  0xF3,  0xB5,
     0x82,  0x65,  0x0E,  0x44,  0xF6,  0xB2,  0x02,  0x5B,
     0x21,  0xA5,  0x93,  0x2E,  0x21,  0xDD,  0xE7,  0x60,
     0xFE,  0x87,  0x0E,  0xDC,  0x42,  0x2F,  0xED,  0xC4,
     0x37,  0xC1,  0x48,  0x3C,  0x28,  0xC7,  0x90,  0x41,
     0x19,  0xB1,  0xE2,  0x94,  0x14,  0x4A,  0x1C,  0xB3,
     0xCA,  0xE4,  0xFC,  0xAE,  0xD8,  0x1E,  0x27,  0xC7,
     0x53,  0x7C,  0x8E,  0xD1,  0x90,  0x7F,  0x1B,  0xDD,
     0x62,  0x02,  0x38,  0xA6,  0x39,  0xF8,  0xAF,  0x57,
     0x00,  0x27,  0xE4,  0x3F,  0x0C,  0xDC,  0xF8,  0x77,
     0x43,  0xB2,  0x81,  0x73,  0xE5,  0x0B,  0x11,  0x6A,
     0x0C,  0x4D,  0x50,  0x11,  0xA4,  0x0D,  0x91,  0x68,
     0x9E,  0xB5,  0xA7,  0xD3,  0xE8,  0x25,  0x0A,  0xA5,
     0xF9,  0xED,  0xE0,  0x97,  0x33,  0x8F,  0x76,  0x5C,
     0xCD,  0xB7,  0x23,  0x06,  0x27,  0x70,  0x2F,  0x5A,
     0xE5,  0xA1,  0x08,  0xC9,  0x2A,  0x73,  0x2A,  0x63,
     0xEF,  0x41,  0xB9,  0x73,  0x70,  0x23,  0x7E,  0x2F,
     0xAB,  0x2A,  0xFE,  0x72,  0x3A,  0x6C,  0xF9,  0xE8,
     0x94,  0x3D,  0xD8,  0xB2,  0xB5,  0x6D,  0x4B,  0xF5,
     0x53,  0xBC,  0xC3,  0x2C,  0xD4,  0x1D,  0xB8,  0x2E,
     0x5D,  0x6D,  0x56,  0x3B,  0x30,  0xB1,  0x7C,  0xDE,
     0x70,  0xA2,  0x18,  0x60,  0xE9,  0xDC,  0x12,  0x9F,
     0xC5,  0xBF,  0x8C,  0x24,  0x5F,  0x66,  0x5F,  0x93,
     0x28,  0x33,  0x0C,  0xE3,  0x75,  0x54,  0x45,  0x9D,
     0x6A,  0x60,  0xFC,  0x91,  0x19,  0x39,  0x97,  0xEC,
     0xF0,  0x23,  0xD0,  0xFE,  0xB6,  0xD4,  0x6F,  0x9E,
     0x92,  0xF4,  0xED,  0xCD,  0x31,  0x36,  0xE5,  0x42,
     0x4A,  0xB5,  0x09,  0xE4,  0x2E,  0x22,  0xD0,  0xC1,
     0xC0,  0x58,  0xB2,  0x3D,  0x5F,  0x1F,  0xC0,  0x96,
     0x09,  0x4F,  0xD0,  0xEE,  0xB5,  0x21,  0x92,  0x2D,
     0x9D,  0xD1,  0x94,  0x82,  0x70,  0x32,  0x16,  0x15,
     0xEA,  0x75,  0x06,  0x25,  0x48,  0xAC,  0x17,  0xDF,
     0x82,  0x95,  0x81,  0x9E,  0xE6,  0x0E,  0x9E,  0xA4,
     0x82,  0x3C,  0xC6,  0xE8,  0x68,  0x65,  0xC7,  0xBD,
     0x5B,  0x04,  0x00,  0xDF,  0x96,  0x6C,  0x89,  0x85,
     0xD4,  0xA2,  0xFE,  0xB6,  0x19,  0x6E,  0xE3,  0x14,
     0xD2,  0x08,  0x10,  0xDD,  0xE9,  0xDC,  0x93,  0x42,
     0x11,  0x6E,  0x94,  0x76,  0xEC,  0x37,  0xE6,  0xD5,
     0xD6,  0x6C,  0xE8,  0xE9,  0xC8,  0xC8,  0xB1,  0x62,
     0x49,  0x81,  0x92,  0xB8,  0x97,  0xE4,  0x5A,  0x32,
     0x00,  0x2E,  0xCB,  0x9C,  0x63,  0x65,  0xDF,  0xA7,
     0x26,  0xAF,  0x35,  0xCC,  0x06,  0xF1,  0x54,  0x7A,
     0x69,  0x03,  0x5C,  0x76,  0x4D,  0x51,  0xCE,  0xD0,
     0xDF,  0x9E,  0xF7,  0xB4,  0xFA,  0x17,  0xD3,  0x4A,
     0xE7,  0xAC,  0x25,  0xB3,  0xAF,  0x05,  0x71,  0x13,
     0x0A,  0x35,  0xD1,  0x3F,  0x83,  0xA0,  0xCF,  0x7C,
     0xDE,  0xC5,  0x44,  0xCF,  0x7C,  0x63,  0xB5,  0xE0,
     0xF9,  0x93,  0x5C,  0xD8,  0x05,  0x9A,  0xB3,  0x63,
     0xF6,  0x26,  0x6B,  0x5D,  0xED,  0x06,  0xD1,  0x96,
     0xC6,  0xAB,  0x91,  0xDF,  0x82,  0xAA,  0x79,  0x3C
};

unsigned char  out_data_c[COLS];

//unsigned char  out_data_c_expected[COLS] = { 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 64, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 0, 0, 128, 0, 0, 0, 0, 0, 32, 8, 0, 0, 0, 0, 64, 0, 0, 0, 16, 2, 0, 128, 64, 0, 0, 0, 0, 0, 0, 128, 0, 32, 0, 0, 0, 8, 0, 0, 128, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 2, 4, 0, 0, 1, 4, 192, 10, 0, 0, 32, 0, 0, 0, 0, 32, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 64, 0, 0, 0, 16, 0, 0, 0, 5, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 64, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 12, 0, 32, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 24, 4, 26, 0, 1 };

// MicroBlaze, LEON3
unsigned char  out_data_c_expected[COLS] = { 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63, 255, 255, 255, 63 };


const char  mask[9] =
{
    -1, 1, -1,
     1, 1,  1,
    -1, 1, -1
};


int main(int argc, char** argv) {

        #pragma monitor start
        #pragma kernel
	IMG_erode_bin_c(in_data, out_data_c, mask, COLS);
        #pragma monitor stop

	if (argc > 42 && ! strcmp(argv[0], ""))	printf("%u", (unsigned int) out_data_c[COLS-1]);


	int i;

	for(i=0; i < COLS; i++) {
			if(out_data_c[i] != out_data_c_expected[i]) {
					return 1;
			}
	}


	return 10;

}
