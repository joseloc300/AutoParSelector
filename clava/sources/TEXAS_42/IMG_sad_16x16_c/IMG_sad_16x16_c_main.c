#include <stdio.h>
#include <string.h>

unsigned IMG_sad_16x16_c
(
    const unsigned char * srcImg,
    const unsigned char * refImg,
    int pitch
);

/* ======================================================================== */
/*  Constant dataset.                                                       */
/* ======================================================================== */
#define pitch    (16)


/* ======================================================================== */
/*  Initialize arrays with random test data.                                */
/* ======================================================================== */
const unsigned char  src_data[256] =
{
     0xCE,  0x8F,  0x84,  0x01,  0x91,  0x2F,  0x96,  0x1C,
     0x06,  0xFA,  0x06,  0x15,  0x5E,  0x0E,  0x91,  0xAD,
     0x10,  0x78,  0x04,  0xE2,  0xEA,  0xDF,  0xA0,  0x74,
     0xCD,  0x5E,  0x1D,  0x4A,  0x70,  0xCA,  0x55,  0xFE,
     0x62,  0xE9,  0x74,  0xEA,  0x53,  0xE9,  0x8E,  0x3B,
     0xB5,  0x33,  0xCC,  0xE9,  0xB3,  0x7F,  0xB1,  0x79,
     0xEE,  0x20,  0x7B,  0xB6,  0xBD,  0xEE,  0x49,  0x45,
     0xA5,  0x58,  0xF0,  0xC5,  0xCA,  0x59,  0xB5,  0xFD,
     0xDA,  0x7A,  0xE4,  0xEE,  0x58,  0x01,  0x75,  0xE7,
     0x1C,  0xF3,  0x13,  0xE4,  0x04,  0xC7,  0x8C,  0x67,
     0xD0,  0xD4,  0x3E,  0x0E,  0x96,  0x23,  0xA7,  0x22,
     0xE7,  0x0B,  0x86,  0x37,  0x8D,  0x2F,  0x2A,  0x24,
     0xD7,  0x19,  0xA3,  0xB2,  0x69,  0x01,  0x47,  0x80,
     0x38,  0x4D,  0x63,  0xF1,  0x5E,  0xA9,  0xFD,  0x82,
     0xE7,  0x01,  0xDD,  0x2C,  0x1F,  0xA3,  0x9F,  0x4F,
     0x29,  0x19,  0x35,  0x4B,  0xCF,  0x81,  0xC5,  0x96,
     0xEF,  0x4D,  0x5F,  0x39,  0x35,  0xBB,  0xCD,  0x3D,
     0xF1,  0x4B,  0x64,  0xBD,  0xDF,  0x6D,  0xEF,  0xDB,
     0x1B,  0xFB,  0xB2,  0x15,  0xE4,  0xCB,  0xB4,  0x8E,
     0x54,  0x50,  0xAE,  0x04,  0x90,  0xAD,  0x1A,  0xDF,
     0x9F,  0x45,  0x5F,  0x4E,  0x71,  0x66,  0x06,  0x6D,
     0x1A,  0x88,  0x41,  0x71,  0xE5,  0x91,  0x9C,  0xA6,
     0x61,  0x60,  0x72,  0x0B,  0x5B,  0x1C,  0x14,  0x12,
     0x95,  0xFE,  0x49,  0x63,  0x6D,  0x0D,  0x34,  0x97,
     0x28,  0x50,  0x11,  0xB0,  0xF9,  0xFF,  0xFA,  0xBD,
     0x7C,  0xED,  0xFF,  0xDC,  0x5F,  0x60,  0x46,  0x0F,
     0x46,  0x43,  0xD8,  0x0B,  0x35,  0xB1,  0x4C,  0xBE,
     0xA7,  0xAD,  0x89,  0xA7,  0xB1,  0xF3,  0x40,  0x02,
     0xDA,  0x4E,  0xE9,  0x80,  0x62,  0x76,  0x6D,  0x10,
     0x73,  0xFB,  0x47,  0x6D,  0xBB,  0xF3,  0x03,  0x39,
     0x1A,  0x7E,  0xDF,  0xCE,  0x62,  0xBF,  0x1F,  0x3A,
     0xDE,  0xAA,  0x4C,  0xAF,  0x3E,  0x50,  0x86,  0x17
};

const unsigned char  ref_data[3840] =
{
     0x4C,  0x91,  0x1B,  0x73,  0x98,  0x1D,  0x01,  0x73,
     0x95,  0x3B,  0x27,  0x99,  0xEE,  0x1C,  0xFA,  0xF5,
     0x53,  0xCD,  0x15,  0xC4,  0x5C,  0x9D,  0x06,  0x56,
     0x8D,  0xF0,  0x3C,  0xFF,  0xD5,  0x80,  0x22,  0x80,
     0x03,  0xC6,  0xA7,  0x3A,  0xEE,  0xEC,  0x2B,  0x8F,
     0xE7,  0x41,  0x4A,  0x1E,  0x20,  0xB7,  0xB9,  0xA7,
     0x97,  0x17,  0x81,  0x94,  0x22,  0xD7,  0xD0,  0x43,
     0x29,  0xE5,  0xEB,  0xDF,  0x3F,  0xC7,  0x05,  0x06,
     0x02,  0x48,  0x39,  0xBB,  0x31,  0xFE,  0xA2,  0x3F,
     0x1B,  0xC3,  0x1E,  0xAB,  0x68,  0xEA,  0x03,  0xDB,
     0xF5,  0x76,  0xA1,  0xA1,  0x7D,  0xC6,  0xE9,  0x13,
     0xC5,  0x75,  0x2C,  0x28,  0xCD,  0xEA,  0xB9,  0xF0,
     0xCD,  0x75,  0x61,  0x7B,  0x32,  0xE8,  0xA3,  0xAB,
     0x65,  0x51,  0x78,  0x5F,  0x35,  0xC6,  0xA8,  0x0B,
     0xCC,  0x77,  0xF8,  0x09,  0xFD,  0x34,  0xCA,  0x14,
     0x58,  0x00,  0x17,  0x23,  0xB9,  0x7F,  0x68,  0xCF,
     0x4F,  0x7D,  0x9F,  0xD7,  0x58,  0x4D,  0xA4,  0x77,
     0x4F,  0x24,  0x27,  0xB7,  0xCA,  0xD7,  0xE1,  0x19,
     0xDB,  0x12,  0xB7,  0xA8,  0x2B,  0x8A,  0x07,  0x79,
     0x4B,  0xA6,  0x51,  0x16,  0xDF,  0x5D,  0xB2,  0x44,
     0x39,  0x20,  0xBE,  0x72,  0xA1,  0x3D,  0xE3,  0x87,
     0x29,  0x86,  0xFA,  0x52,  0x55,  0x1B,  0xB5,  0xF4,
     0x03,  0xD7,  0xF2,  0x9D,  0x8A,  0x02,  0x87,  0xCF,
     0xDB,  0x6C,  0xF6,  0xE9,  0x5E,  0xB7,  0xB8,  0x2E,
     0x47,  0xF3,  0x12,  0x6E,  0xAA,  0xE4,  0x64,  0xCE,
     0x78,  0x4B,  0xC6,  0x18,  0x2B,  0xE2,  0xD6,  0xF9,
     0x35,  0xE1,  0xF8,  0xCF,  0xC0,  0x6B,  0x5E,  0xCD,
     0xC7,  0xB8,  0xC9,  0x70,  0x88,  0x62,  0xFB,  0xCC,
     0xF2,  0x91,  0xF9,  0xDE,  0x30,  0xF8,  0xCC,  0xB1,
     0xFD,  0xF2,  0xD3,  0x49,  0x9F,  0x17,  0x96,  0x76,
     0x07,  0xEF,  0x32,  0xF3,  0x90,  0xE9,  0xD0,  0xEA,
     0xC3,  0xA3,  0x12,  0xD6,  0xB6,  0xC2,  0x76,  0x53,
     0x0E,  0x4D,  0x43,  0x01,  0x90,  0x78,  0xA4,  0x7A,
     0xC2,  0xD4,  0x48,  0xE5,  0xF0,  0x73,  0x52,  0xEC,
     0x80,  0x48,  0x1E,  0x92,  0xFD,  0x4C,  0xF5,  0x4C,
     0x44,  0xAD,  0xAC,  0xA6,  0x7E,  0xE0,  0x84,  0x6E,
     0xDE,  0xDD,  0xDF,  0xB3,  0xC3,  0x2E,  0x81,  0x5F,
     0xB7,  0xED,  0x07,  0xF5,  0xD7,  0x16,  0xE9,  0x78,
     0x89,  0xCD,  0xDF,  0x93,  0x40,  0x63,  0x82,  0x83,
     0x0F,  0x3F,  0xD3,  0xFC,  0xBB,  0x2C,  0x03,  0x3A,
     0x08,  0x77,  0x6A,  0xAD,  0x53,  0x82,  0x95,  0x96,
     0x5C,  0xC5,  0x7F,  0x3E,  0x3A,  0x0D,  0xB5,  0xF2,
     0x8D,  0xC7,  0xDE,  0xC1,  0xE4,  0xE6,  0x3B,  0x99,
     0x1A,  0x91,  0x1B,  0x4D,  0xFB,  0x62,  0x47,  0x27,
     0xFE,  0xEF,  0x08,  0xF7,  0xD7,  0x0A,  0x19,  0xFC,
     0xF9,  0xE5,  0xF8,  0xC8,  0x6F,  0x3E,  0x8A,  0x3E,
     0xCA,  0xF1,  0x08,  0xDA,  0xB3,  0x5B,  0x87,  0xF2,
     0x3B,  0x5E,  0x28,  0x59,  0xC3,  0x1E,  0x45,  0x4F,
     0x55,  0x4A,  0x1D,  0x29,  0x62,  0x6D,  0x2B,  0xCB,
     0xDD,  0x75,  0xC9,  0xD3,  0xA4,  0x41,  0x3D,  0x48,
     0xC8,  0x33,  0x17,  0x94,  0xD4,  0x8C,  0xA6,  0x7A,
     0x20,  0xFB,  0x91,  0xB7,  0x39,  0x95,  0x8F,  0xC5,
     0x82,  0x4A,  0x51,  0xE4,  0x74,  0x10,  0x88,  0xE7,
     0x34,  0x68,  0x0E,  0xA3,  0xE3,  0x99,  0x28,  0x3A,
     0x84,  0xA0,  0x76,  0x2E,  0xA9,  0x06,  0x16,  0xA7,
     0x16,  0x3F,  0x7F,  0x89,  0xA1,  0xFC,  0x8C,  0x46,
     0x8B,  0x75,  0x6F,  0x0C,  0xE4,  0xFF,  0xA9,  0x33,
     0xE0,  0xC2,  0x4E,  0xB0,  0x30,  0xE3,  0x3E,  0x57,
     0xC1,  0x27,  0x3A,  0xF9,  0xE9,  0x1E,  0x90,  0xCA,
     0x1D,  0xCA,  0x7C,  0xCB,  0x3D,  0x11,  0x7C,  0xE9,
     0x22,  0x14,  0x96,  0x4B,  0x50,  0xB1,  0xD1,  0x7E,
     0xE6,  0x7C,  0x95,  0xBC,  0x48,  0x62,  0x1B,  0x15,
     0x0F,  0x7D,  0xAF,  0x6B,  0x7D,  0x00,  0x34,  0x34,
     0xCC,  0x22,  0xF9,  0xC6,  0x0A,  0x51,  0xBB,  0x39,
     0xAB,  0xA7,  0x44,  0x4A,  0x7F,  0x05,  0x6E,  0x7F,
     0xE5,  0x7B,  0x87,  0x43,  0x7C,  0x7C,  0x9D,  0xDE,
     0xF4,  0xC2,  0xF7,  0x29,  0x93,  0x48,  0x55,  0xAB,
     0x7A,  0x28,  0x09,  0x23,  0xD8,  0x59,  0xCD,  0x3F,
     0xB6,  0x5B,  0xAE,  0x35,  0x46,  0x13,  0x72,  0x60,
     0x1C,  0x26,  0xE1,  0xCC,  0x26,  0x9A,  0x7E,  0x02,
     0xCE,  0x5F,  0x45,  0x99,  0x6C,  0xA5,  0x6C,  0xAB,
     0x3B,  0x69,  0xD4,  0x1D,  0x3C,  0xEA,  0xAE,  0x06,
     0x5B,  0xF1,  0xF4,  0xF7,  0x68,  0x3E,  0x04,  0x5D,
     0x75,  0x10,  0xFB,  0xA5,  0x3B,  0xFA,  0x7E,  0x63,
     0xD3,  0x9C,  0x25,  0x74,  0x75,  0x00,  0xB1,  0x02,
     0x2C,  0xC5,  0x38,  0x5D,  0xEA,  0x1E,  0xE4,  0x00,
     0x10,  0xA0,  0xA4,  0xC2,  0xF8,  0x19,  0x17,  0xF2,
     0x2F,  0x3D,  0xB0,  0x73,  0x84,  0xDE,  0x81,  0x46,
     0xD2,  0x6C,  0x5F,  0xD9,  0x09,  0xB2,  0xD3,  0x30,
     0x81,  0xF3,  0x26,  0xE9,  0xDD,  0x3E,  0xDC,  0xED,
     0x5E,  0x72,  0x1C,  0x4A,  0xAE,  0x88,  0x6E,  0x1B,
     0x7D,  0x83,  0x46,  0x29,  0xE4,  0xAB,  0x42,  0xDF,
     0x0C,  0xFA,  0xEB,  0x5A,  0x83,  0x3C,  0x60,  0x27,
     0xEC,  0x59,  0x2F,  0xBF,  0xEA,  0xC5,  0x09,  0xA9,
     0x11,  0x99,  0x31,  0x4E,  0xC4,  0xB4,  0x7F,  0x25,
     0xC9,  0x7C,  0xCA,  0xE3,  0x46,  0x75,  0x01,  0x07,
     0xC9,  0x69,  0x80,  0x99,  0xFD,  0x16,  0x45,  0xCD,
     0xCB,  0xBF,  0xCA,  0x8B,  0x24,  0x0E,  0x44,  0x6D,
     0x4A,  0x3B,  0xBD,  0xDA,  0xC6,  0x47,  0xB9,  0x8F,
     0xF1,  0x99,  0x5A,  0x1D,  0xA7,  0x69,  0xBA,  0xA9,
     0x12,  0x4B,  0x2F,  0xD4,  0x59,  0xF0,  0xFC,  0xDD,
     0xA2,  0x61,  0xCC,  0x05,  0xA8,  0x27,  0xF8,  0x01,
     0x05,  0x3B,  0x76,  0xCE,  0xE9,  0x6A,  0xB4,  0x7C,
     0x20,  0xCB,  0xE9,  0xCB,  0xB9,  0xA9,  0x5F,  0x5A,
     0x1E,  0x59,  0x99,  0x04,  0x00,  0x17,  0xDF,  0x8A,
     0x67,  0xCC,  0x96,  0x65,  0x3B,  0x4E,  0x8E,  0x60,
     0x85,  0x77,  0x99,  0x23,  0x57,  0x15,  0xC6,  0x39,
     0xAB,  0x44,  0xFD,  0xD9,  0xA5,  0x0F,  0x9A,  0xA7,
     0xE5,  0xDB,  0x3A,  0xEC,  0xDE,  0x49,  0x19,  0x84,
     0x17,  0x22,  0x77,  0x83,  0x51,  0x98,  0xA0,  0x4D,
     0x38,  0x02,  0xF7,  0x74,  0xF9,  0x22,  0x75,  0x50,
     0x7D,  0xD3,  0xD5,  0x80,  0x6C,  0x57,  0x93,  0x90,
     0x68,  0x50,  0x49,  0xB5,  0x1C,  0xA9,  0xC4,  0xC0,
     0x16,  0x32,  0xC8,  0x12,  0xE3,  0x40,  0x71,  0x58,
     0x79,  0xD5,  0xC6,  0x46,  0x46,  0xAC,  0x5A,  0xAE,
     0x8F,  0x57,  0x92,  0x08,  0x60,  0x30,  0x36,  0xB8,
     0x1C,  0xCB,  0xA9,  0x87,  0x1A,  0x28,  0xA7,  0x94,
     0xE5,  0xE5,  0x34,  0x7B,  0xB3,  0x24,  0x2C,  0xD4,
     0xD0,  0x78,  0xD6,  0x91,  0x82,  0x33,  0xE1,  0x57,
     0xF3,  0xCB,  0xC9,  0x63,  0x42,  0xC1,  0x87,  0xC2,
     0x0E,  0x8A,  0x72,  0xA4,  0x22,  0x09,  0x17,  0xAE,
     0x99,  0x8A,  0xBE,  0xDF,  0x5A,  0xDF,  0x60,  0x40,
     0x0F,  0x2A,  0x8E,  0x0A,  0x06,  0x0F,  0x6E,  0x34,
     0xE4,  0x71,  0x16,  0x2E,  0x73,  0x07,  0x73,  0x63,
     0x2A,  0x49,  0x90,  0x80,  0x57,  0xDD,  0x8A,  0x4C,
     0xA9,  0x76,  0xF6,  0xB4,  0xC4,  0x1B,  0x4B,  0x9B,
     0xE7,  0xF9,  0x4B,  0xB2,  0xFB,  0xA8,  0x6C,  0x66,
     0x68,  0x89,  0x17,  0x8E,  0xC9,  0xA6,  0xAB,  0xA0,
     0x33,  0xC6,  0xF8,  0x5E,  0x68,  0x99,  0x3A,  0x52,
     0x82,  0x85,  0xE5,  0x86,  0x7E,  0x7D,  0x66,  0x33,
     0x64,  0x5C,  0x88,  0x1E,  0x1D,  0xF7,  0xBB,  0xB1,
     0xFC,  0x30,  0x66,  0x73,  0x81,  0xA7,  0x00,  0xC4,
     0xE9,  0x08,  0xEC,  0xF0,  0x79,  0x24,  0x81,  0xAB,
     0x6E,  0xE6,  0x30,  0x57,  0x64,  0xCB,  0xB4,  0x5E,
     0xDD,  0xC0,  0x61,  0x06,  0xF0,  0xD2,  0x08,  0x79,
     0xD1,  0xCF,  0x06,  0xBB,  0xCA,  0x91,  0xCA,  0x7D,
     0xDB,  0xD7,  0xD7,  0x70,  0xCA,  0xFC,  0x4C,  0x7B,
     0x45,  0xD1,  0xEB,  0x24,  0x2F,  0xA9,  0x57,  0xA1,
     0xC6,  0x8B,  0xEF,  0x98,  0xF7,  0x7F,  0x88,  0xCB,
     0x10,  0xA1,  0xBE,  0xAE,  0x6F,  0x77,  0xCB,  0xB8,
     0x63,  0xF5,  0x45,  0xC4,  0x9F,  0x5D,  0x3C,  0xA4,
     0x66,  0x88,  0xA8,  0x07,  0x5F,  0x8C,  0xEC,  0xD2,
     0xF8,  0x28,  0xEF,  0xFE,  0x6E,  0x10,  0x97,  0xF9,
     0xB3,  0xDC,  0xF7,  0x6D,  0x20,  0x6C,  0x21,  0x9D,
     0x54,  0x83,  0x64,  0x1A,  0xE0,  0x70,  0xE7,  0x1C,
     0x88,  0x38,  0x34,  0x78,  0xFA,  0x50,  0x2B,  0xA3,
     0xE6,  0x73,  0x43,  0xBA,  0xF0,  0x0D,  0xBB,  0x4C,
     0x61,  0xE9,  0x7E,  0xAC,  0xD5,  0xDD,  0xB5,  0x4D,
     0xD5,  0x30,  0xA0,  0x8D,  0xFA,  0x0A,  0xC0,  0x92,
     0xD0,  0x31,  0x8B,  0x38,  0xB6,  0xFB,  0x4D,  0x21,
     0x38,  0x38,  0xCF,  0x2C,  0xB1,  0xDC,  0x94,  0x3F,
     0xE2,  0x3B,  0xC9,  0x5D,  0xCD,  0x57,  0xB6,  0xC2,
     0xE0,  0x93,  0xDA,  0x43,  0x7C,  0x6A,  0x21,  0xE7,
     0xB4,  0xF9,  0x88,  0x54,  0xF4,  0x24,  0xB0,  0xA2,
     0x58,  0x17,  0x1C,  0xF0,  0x9F,  0x7F,  0x24,  0xCD,
     0x97,  0x3C,  0x2A,  0xB8,  0xB2,  0x35,  0x8F,  0x6D,
     0xFD,  0x44,  0xA5,  0x92,  0xEF,  0x89,  0x01,  0xF3,
     0x31,  0xB7,  0xBC,  0xCC,  0x21,  0x85,  0x4D,  0xA7,
     0x6D,  0x6A,  0x5F,  0x67,  0x0F,  0x12,  0x08,  0x58,
     0x83,  0xD3,  0x77,  0x1F,  0x4D,  0xB8,  0xEF,  0xF9,
     0x9B,  0x2E,  0x28,  0xA6,  0x6B,  0x70,  0xC7,  0x1F,
     0xC5,  0x71,  0x09,  0x5E,  0xE8,  0x3A,  0x67,  0x2C,
     0x43,  0xA6,  0x49,  0x13,  0x62,  0x96,  0x0E,  0x94,
     0x64,  0x06,  0xAD,  0x69,  0x63,  0x04,  0x4A,  0xCB,
     0xA1,  0x9A,  0x08,  0x22,  0x6E,  0x11,  0xC5,  0x53,
     0xBF,  0xB0,  0x68,  0xEB,  0xD2,  0x24,  0x04,  0xED,
     0x97,  0xAE,  0x36,  0x29,  0x25,  0x82,  0xB4,  0x0F,
     0x4E,  0x32,  0xFD,  0x0C,  0x29,  0x8F,  0x66,  0xA8,
     0xA5,  0x71,  0xF5,  0x47,  0x69,  0x36,  0xDB,  0xAD,
     0x56,  0xF0,  0x65,  0x05,  0xB3,  0xF3,  0xAE,  0x17,
     0x60,  0xA5,  0x28,  0xE9,  0x2E,  0xA2,  0x0A,  0xC0,
     0x62,  0x5D,  0xDC,  0xA6,  0xD5,  0x6C,  0x6E,  0x12,
     0x55,  0x37,  0x38,  0x20,  0x91,  0xEA,  0xBD,  0x8D,
     0x12,  0x73,  0xD8,  0x19,  0x7C,  0x64,  0xF1,  0xEC,
     0x76,  0xB5,  0x1F,  0x35,  0x3F,  0xB7,  0x7E,  0x2E,
     0x08,  0x23,  0x7A,  0x6C,  0xCE,  0x0B,  0x01,  0xD9,
     0x89,  0x43,  0xEF,  0x23,  0x65,  0xF8,  0x57,  0x76,
     0xFA,  0xE2,  0x46,  0xBB,  0xEC,  0x1B,  0x27,  0xDB,
     0x2F,  0x54,  0x41,  0xE9,  0x97,  0x6A,  0x02,  0xA8,
     0x36,  0xB3,  0x38,  0xF4,  0xF8,  0xDC,  0xD3,  0x51,
     0x84,  0xAC,  0x45,  0xDE,  0x78,  0xEA,  0xE6,  0x37,
     0x18,  0x55,  0x84,  0x94,  0x96,  0xAB,  0x02,  0x84,
     0x66,  0x79,  0x57,  0x75,  0x0C,  0x00,  0x24,  0x07,
     0x4A,  0x77,  0x90,  0xE2,  0x9E,  0x5B,  0x47,  0xCA,
     0xE4,  0x70,  0x54,  0x1F,  0xF7,  0x17,  0x25,  0xF0,
     0xC5,  0x26,  0x04,  0x70,  0xCA,  0x4C,  0x5D,  0x30,
     0x6F,  0x56,  0x21,  0x37,  0x1E,  0x5B,  0x81,  0x87,
     0xEC,  0xC0,  0xF0,  0xFA,  0x77,  0x07,  0x96,  0xA9,
     0xBB,  0x54,  0x0C,  0x23,  0x66,  0x33,  0x2A,  0x5C,
     0x41,  0x29,  0x63,  0xEB,  0xDC,  0xB3,  0xD7,  0x3B,
     0x78,  0x00,  0x00,  0x1A,  0x85,  0xC1,  0x26,  0x33,
     0x7E,  0x1A,  0x0B,  0x0E,  0x48,  0xC2,  0xF7,  0xBB,
     0x54,  0xFE,  0xBB,  0x3D,  0x22,  0xF1,  0x61,  0xF7,
     0x1D,  0xB2,  0x9C,  0x47,  0x43,  0x96,  0xA9,  0xEC,
     0xD8,  0x98,  0x72,  0xEC,  0xBE,  0xF2,  0x49,  0x5B,
     0x9A,  0xB3,  0x22,  0x50,  0xA9,  0x16,  0x4D,  0x38,
     0x2C,  0xC5,  0xA1,  0x86,  0x1A,  0x28,  0x36,  0x6F,
     0xEE,  0x00,  0x81,  0xD7,  0x19,  0x58,  0x4C,  0x3F,
     0xC9,  0x67,  0xE3,  0x01,  0x17,  0xFC,  0xF6,  0xA0,
     0x1E,  0x5C,  0xBE,  0x7F,  0x5A,  0xEA,  0xDE,  0xF6,
     0x95,  0xC2,  0x1F,  0x0A,  0x4A,  0x01,  0xD7,  0xE4,
     0xCE,  0x72,  0xDC,  0x95,  0x77,  0x5E,  0x5C,  0x21,
     0x12,  0x79,  0x58,  0x90,  0xB6,  0x66,  0x18,  0x0E,
     0x41,  0xA3,  0x60,  0x83,  0xC7,  0x19,  0x83,  0x4B,
     0x7E,  0x8D,  0xF8,  0xF8,  0x6F,  0xAF,  0xA4,  0x82,
     0x29,  0x27,  0xCD,  0x55,  0x29,  0x90,  0x40,  0x9C,
     0x19,  0x29,  0x66,  0x52,  0x18,  0x19,  0x67,  0xCD,
     0x38,  0x85,  0xA9,  0xD6,  0x15,  0x77,  0xEE,  0x22,
     0xF4,  0x3E,  0x2A,  0x51,  0x77,  0x34,  0xC0,  0xDF,
     0x58,  0x65,  0xDA,  0x16,  0x4D,  0x81,  0x20,  0x75,
     0xEC,  0x30,  0x09,  0xE2,  0x98,  0x95,  0xC0,  0xDA,
     0xEA,  0x3D,  0x71,  0x62,  0x5F,  0xBD,  0x66,  0xC5,
     0xCB,  0x24,  0xD4,  0x9A,  0x36,  0xA7,  0x4C,  0xBA,
     0x8B,  0x68,  0x2F,  0xFE,  0x3C,  0xBA,  0x96,  0xBD,
     0xA7,  0xA4,  0xC7,  0x60,  0x6A,  0xF5,  0x5A,  0x57,
     0x2B,  0xA4,  0x59,  0x30,  0x82,  0xDD,  0x93,  0xCE,
     0xF0,  0xB2,  0xBE,  0x0C,  0xCD,  0x8A,  0xC1,  0x78,
     0x7D,  0xFF,  0xA2,  0x68,  0x5E,  0xBA,  0xA5,  0xC9,
     0xD8,  0x76,  0xAA,  0xD4,  0x94,  0x2F,  0x6D,  0xEE,
     0x0B,  0xB3,  0x50,  0x85,  0x0B,  0x40,  0xC5,  0xD3,
     0xF1,  0x28,  0xEB,  0xCC,  0x54,  0xA5,  0xEF,  0x05,
     0x76,  0x88,  0xD8,  0xA2,  0x58,  0x15,  0x7E,  0x2A,
     0x44,  0xE3,  0x95,  0xCF,  0x6B,  0x2D,  0xAC,  0xBB,
     0xAA,  0x9A,  0x8C,  0xD4,  0xC6,  0x70,  0x4A,  0x3D,
     0x28,  0x74,  0xC2,  0x97,  0x46,  0xEE,  0x29,  0x79,
     0x17,  0xC4,  0x2E,  0xC2,  0x14,  0x43,  0x80,  0x0B,
     0xA0,  0x67,  0x65,  0xD1,  0x03,  0x20,  0x38,  0x54,
     0x23,  0xED,  0x75,  0x19,  0x61,  0xA2,  0x41,  0xD4,
     0x19,  0xD0,  0x52,  0x7F,  0x1D,  0xF3,  0x0B,  0x10,
     0x7C,  0x04,  0xD3,  0x21,  0x26,  0x8F,  0xFF,  0x84,
     0xB1,  0xA1,  0x6D,  0xEF,  0x24,  0xA7,  0x65,  0x59,
     0xF9,  0x70,  0x11,  0x14,  0xB6,  0xB6,  0x78,  0x76,
     0x7D,  0x8C,  0x26,  0x1C,  0xB6,  0x54,  0x70,  0xFC,
     0x21,  0x2D,  0x93,  0xF2,  0x11,  0xBA,  0x56,  0x5D,
     0x60,  0xB2,  0xC7,  0x3A,  0x36,  0x3B,  0xF5,  0x17,
     0x9B,  0xF6,  0x52,  0xE4,  0x2A,  0x17,  0xC8,  0x87,
     0x78,  0xA6,  0x2C,  0xC5,  0xEF,  0xC4,  0x00,  0x7A,
     0x18,  0x17,  0xDB,  0x83,  0xEB,  0xC0,  0xB6,  0xC2,
     0x33,  0x96,  0xF3,  0x62,  0xB0,  0x61,  0x27,  0xBD,
     0x6F,  0xDE,  0xEC,  0x80,  0xAC,  0x01,  0x7E,  0x8D,
     0x7E,  0x86,  0x36,  0x49,  0x00,  0x10,  0xF4,  0xC3,
     0x0D,  0xC3,  0x75,  0x92,  0xD4,  0x44,  0x41,  0x6B,
     0xBC,  0xF8,  0x58,  0x28,  0x7E,  0x29,  0x3E,  0xA4,
     0xE9,  0xAA,  0x15,  0x96,  0xD5,  0xBE,  0x44,  0x7C,
     0x62,  0x80,  0x8D,  0xA6,  0x12,  0xB4,  0x6B,  0x4A,
     0x97,  0xF0,  0x6C,  0x51,  0xD6,  0x2F,  0xE3,  0xAF,
     0x6D,  0xFB,  0x0C,  0xFC,  0xEB,  0x77,  0xDA,  0x23,
     0x37,  0x23,  0xBA,  0x47,  0x9C,  0x3A,  0x11,  0x36,
     0x13,  0x8A,  0x1C,  0x42,  0x95,  0x75,  0xF8,  0xC9,
     0x53,  0x80,  0xB9,  0x81,  0x9A,  0xEE,  0x77,  0x0E,
     0x5C,  0x7B,  0x79,  0x70,  0x7F,  0x90,  0xC5,  0x77,
     0xEE,  0xAA,  0xAF,  0x49,  0x2F,  0x92,  0x9A,  0xBC,
     0x89,  0xB0,  0xAA,  0x2B,  0xD0,  0x84,  0xC4,  0xA5,
     0x4E,  0x3B,  0x13,  0x31,  0x7E,  0xDA,  0xAB,  0xA6,
     0x76,  0x4F,  0x57,  0xE1,  0x61,  0x73,  0x9C,  0x3C,
     0x4F,  0x0E,  0x4F,  0xE4,  0x66,  0xFA,  0xD1,  0x8E,
     0x5D,  0xB7,  0xB0,  0xCD,  0x30,  0xAE,  0xE3,  0x1B,
     0x3E,  0x6D,  0x7B,  0xA0,  0x85,  0x52,  0x2D,  0x23,
     0x9F,  0xFD,  0x77,  0xE2,  0xB4,  0x7F,  0xE1,  0xF8,
     0x94,  0x88,  0x11,  0x53,  0x51,  0x9E,  0xDB,  0xBB,
     0x8B,  0x84,  0x40,  0xB5,  0xD4,  0x12,  0x40,  0xC5,
     0x06,  0xD8,  0x01,  0xBC,  0xA7,  0xF2,  0x9E,  0x91,
     0x38,  0x65,  0xE9,  0xEC,  0x73,  0xD6,  0xF0,  0x32,
     0xC9,  0x4E,  0x99,  0x07,  0x69,  0xDB,  0x1C,  0x28,
     0xE7,  0xB5,  0x80,  0xC9,  0xD4,  0xDD,  0xBC,  0xEB,
     0x02,  0x83,  0x38,  0xC8,  0xFF,  0x7B,  0xC3,  0xAF,
     0x9A,  0xED,  0xEF,  0xCA,  0x43,  0x1B,  0x5A,  0xA4,
     0xB4,  0x38,  0xB2,  0x4A,  0xD8,  0x70,  0xB9,  0x81,
     0xB9,  0xF7,  0x3E,  0x8B,  0xC6,  0x97,  0xE9,  0x2A,
     0xAE,  0xDF,  0xE0,  0x9A,  0x4C,  0xE2,  0x82,  0x25,
     0xFE,  0xA3,  0x49,  0x4D,  0xCA,  0xE4,  0x29,  0xF1,
     0x4B,  0x05,  0xC7,  0xBF,  0x5F,  0x0B,  0x29,  0x5F,
     0x09,  0x91,  0x18,  0xDA,  0x04,  0x70,  0xEB,  0xE6,
     0x07,  0xC8,  0x48,  0x04,  0x5F,  0x07,  0x23,  0x37,
     0x4A,  0xCF,  0x5D,  0xB4,  0x02,  0x89,  0x7C,  0x75,
     0x32,  0xBA,  0x4E,  0x4E,  0x5A,  0xCC,  0x42,  0x1B,
     0x1C,  0xAC,  0xB4,  0xAE,  0x27,  0x1F,  0x02,  0x12,
     0x4B,  0xDD,  0xE3,  0xD8,  0xD9,  0x94,  0x54,  0x60,
     0x54,  0x8F,  0xA6,  0x69,  0x01,  0xAD,  0x1C,  0xA7,
     0xCE,  0x94,  0xA6,  0xE1,  0x66,  0x21,  0x59,  0xDC,
     0x93,  0xD1,  0x93,  0x5F,  0x44,  0xC2,  0x35,  0xC7,
     0x7A,  0xB2,  0x8E,  0x0D,  0xCC,  0x8B,  0x70,  0x61,
     0x0F,  0xDE,  0x02,  0x69,  0xCA,  0x1A,  0x61,  0x82,
     0x5E,  0x13,  0xEC,  0x93,  0x18,  0x9E,  0xD3,  0x46,
     0xC2,  0x23,  0xF7,  0xE6,  0x6B,  0x4A,  0xA4,  0x2D,
     0x2A,  0x57,  0x17,  0x86,  0xB9,  0xF2,  0x91,  0x45,
     0x1E,  0x96,  0x53,  0xFE,  0x91,  0x5D,  0x0C,  0x8E,
     0xA2,  0xA9,  0x45,  0xB3,  0x5E,  0x2D,  0xD2,  0x6A,
     0xB1,  0xCD,  0x5D,  0xA1,  0xD3,  0xF3,  0xFC,  0x32,
     0x22,  0xC0,  0x69,  0xFD,  0x65,  0x35,  0xE2,  0xCF,
     0x64,  0xF6,  0xF4,  0x33,  0x00,  0xB5,  0x7F,  0xE8,
     0x94,  0x72,  0x1D,  0x2C,  0xF4,  0x31,  0x4F,  0x5C,
     0x44,  0x31,  0x0E,  0x18,  0x59,  0x3F,  0x98,  0x95,
     0x4C,  0x82,  0xF9,  0x8A,  0x15,  0xA7,  0xBB,  0xDC,
     0xEE,  0x10,  0x82,  0x84,  0xFB,  0xC3,  0xD9,  0xEE,
     0xA8,  0x93,  0xD8,  0xD3,  0x78,  0x0F,  0x51,  0x00,
     0x19,  0x46,  0x42,  0x43,  0xA6,  0x03,  0xBB,  0xC3,
     0x6A,  0x62,  0xE5,  0x4E,  0x9F,  0xB1,  0xF7,  0x3A,
     0xD9,  0xBE,  0x7E,  0x5D,  0x5F,  0x84,  0x7B,  0xEC,
     0x1E,  0xA7,  0x7E,  0x10,  0xA2,  0xA0,  0x95,  0x83,
     0x8D,  0xB7,  0x5D,  0xD3,  0xA5,  0xF0,  0x7B,  0x00,
     0x1C,  0x4A,  0x44,  0xC0,  0xDA,  0x27,  0x20,  0x6B,
     0x9D,  0x8A,  0x3F,  0xCC,  0x3A,  0x1C,  0x6E,  0x65,
     0xDA,  0xCD,  0xD8,  0x66,  0x0C,  0x09,  0x45,  0x0E,
     0x70,  0x3A,  0xBA,  0xF6,  0xA7,  0x32,  0xC7,  0x67,
     0xB4,  0x0E,  0x31,  0x13,  0xEE,  0x4B,  0xD0,  0xD6,
     0x48,  0x0C,  0xBB,  0xF8,  0x09,  0xDE,  0x33,  0x64,
     0x53,  0xCC,  0x80,  0x66,  0x23,  0x86,  0x35,  0x1A,
     0xE5,  0x9B,  0x8F,  0x25,  0xCE,  0x8F,  0x1A,  0x36,
     0x58,  0x8D,  0x06,  0x4D,  0xFC,  0xF0,  0xE1,  0xF6,
     0x24,  0x4A,  0xA8,  0xE1,  0x69,  0x24,  0x70,  0x69,
     0x05,  0xCF,  0x6A,  0x79,  0xAC,  0x90,  0x21,  0x0B,
     0xF2,  0x03,  0x6C,  0x68,  0x2F,  0x95,  0x4C,  0xF0,
     0xEB,  0xA9,  0x50,  0x6C,  0xA8,  0x52,  0xDD,  0xEA,
     0x4E,  0x8A,  0x6D,  0xD6,  0xCF,  0x71,  0x0D,  0x5A,
     0xF7,  0x3A,  0x61,  0x10,  0x70,  0xF8,  0x6C,  0x64,
     0x37,  0xFB,  0xC0,  0xA9,  0x36,  0x4B,  0x0E,  0xC8,
     0x4A,  0x9B,  0xF3,  0x3D,  0xFC,  0x0C,  0x35,  0xFF,
     0xB4,  0x1D,  0x69,  0x2F,  0xB4,  0x59,  0x2A,  0x11,
     0xCA,  0x1D,  0xFB,  0xB2,  0x7A,  0x62,  0xF3,  0x42,
     0x5B,  0xA5,  0x01,  0x78,  0xC2,  0xF3,  0x89,  0xE9,
     0x51,  0x1E,  0x0A,  0x46,  0x2A,  0xBE,  0xC0,  0xAD,
     0x06,  0xA5,  0x16,  0xCF,  0xC2,  0xB2,  0x86,  0xF2,
     0xDA,  0xB6,  0x78,  0x6F,  0x78,  0xB2,  0x52,  0x78,
     0x9B,  0xAB,  0xE5,  0xDC,  0x84,  0x4D,  0x9F,  0x03,
     0x31,  0xF5,  0xF4,  0x5B,  0xAA,  0xD7,  0xF5,  0x7C,
     0x0F,  0x56,  0x74,  0xDC,  0x81,  0x8D,  0xBE,  0x17,
     0xE3,  0x96,  0x26,  0x32,  0xBB,  0xD9,  0x3D,  0xE1,
     0x10,  0x6C,  0x29,  0xA2,  0x10,  0xE5,  0x53,  0xAF,
     0x8E,  0x45,  0x1E,  0x4F,  0x39,  0x0E,  0x77,  0x66,
     0xFE,  0xAB,  0x57,  0x43,  0xFA,  0x81,  0x00,  0x8D,
     0xBD,  0xDC,  0xAE,  0x72,  0x4C,  0x90,  0x55,  0x6B,
     0x1C,  0xE3,  0x79,  0xAA,  0x3C,  0xF0,  0xDC,  0x25,
     0xF3,  0x4E,  0xFB,  0x47,  0x2C,  0xFF,  0x5C,  0x1B,
     0x1E,  0x1A,  0xFF,  0x71,  0xDE,  0xAE,  0x92,  0x1F,
     0x9E,  0x05,  0xD8,  0xC6,  0xAB,  0x7C,  0x08,  0x44,
     0x86,  0xC1,  0xE5,  0xB7,  0x2A,  0x2A,  0xD0,  0xC1,
     0x05,  0xFB,  0xB7,  0x32,  0xAB,  0x72,  0xBF,  0xC8,
     0x72,  0x3A,  0x89,  0xD9,  0xA5,  0x1C,  0xD0,  0x33,
     0x6D,  0xD7,  0x48,  0xDA,  0x99,  0x3B,  0xF1,  0xC3,
     0xC3,  0x41,  0x69,  0x42,  0x9D,  0x41,  0x00,  0xD7,
     0x11,  0xC8,  0x17,  0xD6,  0x3F,  0xCB,  0x11,  0xBC,
     0xCB,  0xE8,  0xBA,  0xB4,  0x2D,  0xD2,  0xFF,  0x3B,
     0xA0,  0xEC,  0xAD,  0x48,  0x92,  0xD6,  0x3E,  0x81,
     0xEA,  0x3F,  0x14,  0xC9,  0x03,  0x47,  0x75,  0x5C,
     0x5E,  0x85,  0x1A,  0x0F,  0x47,  0x27,  0xB9,  0x8F,
     0xC0,  0xE2,  0x94,  0x82,  0x54,  0x44,  0x8E,  0x33,
     0x26,  0x48,  0xE5,  0xC9,  0x57,  0x21,  0xA0,  0x1D,
     0xDC,  0xF7,  0x38,  0x26,  0xBE,  0xB4,  0x0C,  0xDC,
     0xD2,  0x73,  0xC2,  0xA2,  0xBF,  0x9A,  0x6D,  0x2E,
     0x19,  0x5D,  0x4D,  0xCF,  0x10,  0x80,  0x3C,  0xB3,
     0xE2,  0x98,  0x96,  0x51,  0x1A,  0x84,  0x3A,  0x39,
     0x0C,  0x0F,  0x3B,  0x5F,  0x2C,  0x62,  0x45,  0x47,
     0x5C,  0x32,  0x9C,  0x36,  0xED,  0x20,  0xD6,  0x5E,
     0x26,  0x0F,  0x0D,  0xB8,  0x85,  0xAB,  0x9F,  0x0D,
     0x31,  0x7A,  0xC0,  0x86,  0xC4,  0xA8,  0x16,  0x21,
     0x82,  0x48,  0x70,  0x7D,  0xE8,  0x24,  0xA6,  0x09,
     0xB6,  0x11,  0x78,  0xED,  0x68,  0xAB,  0xFE,  0x2E,
     0x7E,  0x3C,  0x10,  0xAF,  0x95,  0x41,  0x92,  0x05,
     0xF7,  0x6D,  0xBB,  0x1B,  0xDE,  0x8F,  0xA5,  0xA5,
     0x83,  0x7B,  0x91,  0xE4,  0xCE,  0x5A,  0x51,  0xF3,
     0xE9,  0x33,  0xD5,  0x1F,  0xF8,  0xE7,  0xF7,  0xDA,
     0xB5,  0x2F,  0x8E,  0x02,  0x71,  0xA2,  0x08,  0x92,
     0xC7,  0xCF,  0x35,  0x89,  0x91,  0x9C,  0xB7,  0xA8,
     0x65,  0x3C,  0x47,  0xA6,  0x35,  0x0E,  0x38,  0x7C,
     0x27,  0x0D,  0x86,  0xB3,  0xDB,  0x1B,  0x53,  0xA3,
     0x76,  0xB8,  0x05,  0xA5,  0xA2,  0x66,  0xC6,  0x31,
     0x88,  0x79,  0x9C,  0xAD,  0x45,  0xF8,  0x88,  0xDD,
     0x18,  0xC7,  0x49,  0x5C,  0xE9,  0x25,  0x74,  0xC8,
     0x6A,  0xC0,  0x0C,  0xA1,  0xE1,  0xBF,  0xD8,  0x0D,
     0x8D,  0x1A,  0x50,  0xB4,  0x17,  0xFD,  0x7A,  0x4E,
     0x32,  0x71,  0x82,  0xC3,  0x5C,  0xEA,  0x47,  0x32,
     0xD2,  0x99,  0x8E,  0x04,  0x68,  0x0A,  0x4C,  0x0D,
     0x68,  0xC3,  0x1E,  0x08,  0xE5,  0x2E,  0xFE,  0x1C,
     0x61,  0xF9,  0x14,  0x46,  0xA4,  0x14,  0xC7,  0x39,
     0x14,  0x3D,  0x74,  0x45,  0x90,  0x92,  0xA5,  0x6F,
     0x70,  0x49,  0x19,  0x52,  0xCE,  0x6E,  0x51,  0xBD,
     0x4C,  0x75,  0xF0,  0x71,  0x26,  0x0B,  0xB1,  0xFA,
     0x54,  0xBA,  0x08,  0xF0,  0x8B,  0x46,  0xA0,  0x2C,
     0x28,  0x41,  0xC7,  0x28,  0x62,  0x8F,  0xE9,  0x8A,
     0xFC,  0x1A,  0xD6,  0x14,  0x0E,  0x88,  0x2E,  0x08,
     0xC3,  0xF9,  0xA8,  0xAD,  0xFE,  0xCD,  0xDA,  0x8D,
     0x99,  0xE2,  0x85,  0x92,  0x77,  0x92,  0x9F,  0xE8,
     0xF3,  0xC3,  0xDA,  0x00,  0x32,  0x0F,  0x0B,  0x24,
     0xEE,  0xC0,  0x0A,  0x08,  0xDC,  0x67,  0x93,  0x2E,
     0xD4,  0xF4,  0x88,  0xD5,  0x6E,  0xED,  0x24,  0x84,
     0xED,  0xEE,  0xFE,  0xE6,  0x7E,  0x22,  0xAD,  0x5A,
     0x71,  0x02,  0x4C,  0x7A,  0x71,  0xC9,  0x4F,  0xBC,
     0x85,  0xE6,  0xDE,  0xBD,  0xD9,  0xC8,  0xC9,  0x1F,
     0x00,  0xA1,  0x54,  0xED,  0x13,  0x4B,  0x93,  0x4C,
     0xC8,  0x1D,  0xB8,  0x4C,  0x95,  0xD1,  0xA0,  0xD8,
     0x93,  0xE8,  0xA7,  0xC4,  0x65,  0x59,  0xED,  0x13,
     0xE2,  0xE8,  0x7D,  0xEE,  0x7D,  0xF0,  0x6A,  0xF1,
     0x34,  0xBE,  0x64,  0xA2,  0xEC,  0x2C,  0x5A,  0x83,
     0x71,  0xB9,  0x6A,  0x59,  0x0D,  0x01,  0x2D,  0x6D,
     0xC3,  0xD1,  0x02,  0x4D,  0x20,  0x8F,  0x33,  0x3C,
     0x43,  0x46,  0x3F,  0xE2,  0x3A,  0x1B,  0xC1,  0xA1,
     0x0C,  0xD6,  0xEF,  0xB3,  0x86,  0x6C,  0x6F,  0x4E,
     0x85,  0x5B,  0x22,  0xA5,  0x81,  0x38,  0xD1,  0xB6,
     0xFB,  0xF0,  0x14,  0x69,  0xFB,  0x21,  0xB8,  0xE8,
     0xEE,  0x55,  0x77,  0x4D,  0x61,  0xEA,  0x4A,  0xAA,
     0xD6,  0x5B,  0x0F,  0x72,  0x02,  0x55,  0x71,  0x29,
     0x69,  0xA0,  0x0C,  0x5D,  0xCC,  0x06,  0x0F,  0xCC,
     0xE5,  0xCC,  0x42,  0x57,  0x42,  0x43,  0x0B,  0x54,
     0x4C,  0xA0,  0x4E,  0x30,  0x3F,  0x5B,  0xD9,  0xEC,
     0xF2,  0x21,  0xF2,  0xE8,  0x75,  0xA2,  0x4C,  0xA5,
     0x29,  0xE2,  0x85,  0x14,  0x74,  0xB3,  0x9E,  0xC7,
     0x8A,  0x51,  0x1D,  0x23,  0x73,  0xB3,  0x6B,  0x78,
     0xAF,  0x8B,  0x2F,  0x3B,  0xFB,  0xD3,  0xE9,  0x61,
     0xEF,  0xB1,  0xC6,  0x1E,  0x29,  0x22,  0x1F,  0x98,
     0x44,  0x5A,  0x08,  0x58,  0x16,  0x36,  0xF6,  0x4E,
     0x0E,  0x04,  0xD5,  0xF3,  0x9A,  0xA3,  0x03,  0xD6,
     0x3F,  0xAA,  0x52,  0x32,  0xAF,  0xA4,  0x8A,  0x28,
     0x05,  0xF1,  0xD6,  0x05,  0x3B,  0x94,  0xEE,  0x3D,
     0xE6,  0x51,  0x6C,  0x81,  0x4B,  0xF7,  0xC6,  0xB1,
     0xF8,  0xC4,  0x25,  0x29,  0x4F,  0x0E,  0x2A,  0x8A,
     0xA3,  0x52,  0xD2,  0xCE,  0x3D,  0xA5,  0x7C,  0x59,
     0x4D,  0xA8,  0x62,  0x7B,  0x12,  0x19,  0x97,  0xC4,
     0x0F,  0xB1,  0xFE,  0x47,  0x91,  0xDC,  0xC9,  0x24,
     0x87,  0xAE,  0x31,  0xF0,  0xCE,  0xFD,  0x2E,  0x02,
     0xC7,  0xE3,  0xE2,  0xA4,  0x66,  0x53,  0xE6,  0x36,
     0x53,  0x5C,  0xA4,  0xF5,  0x38,  0xE1,  0x84,  0xD8,
     0x24,  0xA7,  0xE5,  0x9B,  0xAD,  0x16,  0x88,  0x7A,
     0x95,  0x95,  0xD8,  0xA1,  0xA9,  0x2E,  0x19,  0xE7,
     0x56,  0x4D,  0x8F,  0x8A,  0x95,  0xF1,  0x39,  0x21,
     0x6B,  0x08,  0x92,  0x4C,  0x0C,  0x71,  0xB1,  0x8E,
     0x71,  0xAC,  0x67,  0x3B,  0x05,  0x3F,  0x7C,  0xFD,
     0x87,  0x91,  0xFE,  0x94,  0x9D,  0xA5,  0x03,  0xBB,
     0x66,  0x4B,  0x9A,  0xFF,  0xF2,  0x40,  0xB4,  0xEB,
     0x57,  0x25,  0xCD,  0x26,  0xD3,  0x22,  0x69,  0x4E,
     0x14,  0x8E,  0x7A,  0x86,  0x80,  0x41,  0x13,  0xD2,
     0xCA,  0x30,  0x4B,  0xC7,  0x02,  0xA9,  0x6C,  0xA4,
     0xC6,  0xD7,  0xE6,  0x14,  0x29,  0x34,  0x0F,  0xEF,
     0xBD,  0x98,  0x4A,  0x82,  0xA6,  0x38,  0x51,  0x23,
     0xDD,  0xF5,  0x2E,  0x1B,  0x5F,  0x82,  0x22,  0x6C,
     0x59,  0xB4,  0xDA,  0xEC,  0x54,  0xA5,  0x0D,  0xEC,
     0x82,  0x59,  0xFE,  0x50,  0x70,  0x39,  0xCD,  0x81,
     0xE5,  0xF0,  0x34,  0xE7,  0xBE,  0x2F,  0x32,  0x07,
     0x8D,  0xDB,  0x6C,  0xB0,  0x8F,  0xDB,  0x32,  0x97,
     0xD9,  0xF4,  0x62,  0x58,  0x5E,  0x7A,  0xC2,  0xB0,
     0x0D,  0x11,  0x2D,  0x22,  0x50,  0x7A,  0xAC,  0x2F,
     0x44,  0x7F,  0x75,  0xB1,  0x98,  0xAF,  0x0C,  0x97,
     0x11,  0x82,  0x89,  0xA9,  0x01,  0xD8,  0x3B,  0x8B,
     0xB8,  0x4F,  0x55,  0x4F,  0x7F,  0xB1,  0xE5,  0x3D,
     0xA3,  0x33,  0xA8,  0x45,  0xA4,  0xBD,  0xC0,  0x4E,
     0x5C,  0xCB,  0x87,  0x01,  0x7D,  0xA1,  0xED,  0xBC,
     0x1F,  0x5D,  0x2E,  0xFB,  0x77,  0xC1,  0x3D,  0x99,
     0xDF,  0x3E,  0x6F,  0x39,  0x90,  0x1C,  0xBB,  0xA5,
     0x49,  0x43,  0x5A,  0xAF,  0x55,  0x36,  0xB3,  0x5D,
     0x5B,  0xDF,  0xE7,  0xCC,  0xE2,  0xF6,  0x0D,  0xC5,
     0xC9,  0x7A,  0x16,  0xBF,  0x51,  0xEB,  0x42,  0xE2,
     0x6F,  0xFA,  0x3B,  0x2D,  0xE0,  0x5C,  0x64,  0xFC,
     0x05,  0x25,  0xB0,  0x7F,  0x5A,  0xF7,  0xA8,  0xB8,
     0x17,  0xE9,  0xCE,  0x8E,  0x26,  0x94,  0xAD,  0x71,
     0x71,  0xDF,  0x1F,  0x19,  0xD6,  0xD4,  0x45,  0xA1,
     0x10,  0xBC,  0x14,  0x65,  0xD5,  0xD6,  0xCC,  0xD1,
     0xC3,  0x6B,  0x11,  0x69,  0x6D,  0x5B,  0x41,  0x19,
     0xCA,  0xBA,  0x91,  0x1D,  0x32,  0xEA,  0x3E,  0x6A,
     0xBA,  0x60,  0x33,  0xD1,  0x7F,  0x86,  0x7B,  0x8B,
     0x3E,  0x1A,  0x10,  0x12,  0xA3,  0x89,  0x21,  0x4B,
     0x5C,  0x58,  0x58,  0x26,  0x01,  0xFC,  0x51,  0x6D,
     0x20,  0x9F,  0x3F,  0x17,  0x6A,  0xAF,  0x4C,  0xB3,
     0xD5,  0x72,  0x83,  0x32,  0xB5,  0xC6,  0x84,  0xB6,
     0x3B,  0xF8,  0x68,  0x0F,  0xA6,  0x54,  0x56,  0x84,
     0x5B,  0x1B,  0xEE,  0x64,  0xF6,  0xBB,  0xA9,  0x7A,
     0x14,  0x0D,  0xF1,  0x7D,  0x8A,  0x49,  0xAD,  0x82,
     0xCA,  0x74,  0x98,  0xAF,  0x97,  0x85,  0xFF,  0x9B,
     0xF5,  0x95,  0xD5,  0xF6,  0xD1,  0x20,  0x1F,  0x8B,
     0xDC,  0xCA,  0xF5,  0xB7,  0x86,  0x43,  0x94,  0xD5,
     0x18,  0xAA,  0x4F,  0xE5,  0xD4,  0x60,  0x75,  0x15,
     0x22,  0xF3,  0xBD,  0xC1,  0x34,  0x25,  0x08,  0xA1,
     0x8C,  0x2D,  0x59,  0x13,  0xC8,  0x76,  0xFD,  0x9F,
     0x2B,  0x8C,  0x0A,  0x0B,  0xF3,  0xA1,  0x70,  0xA8,
     0x08,  0x30,  0xC7,  0xD0,  0x21,  0x15,  0x31,  0xE9,
     0x86,  0x61,  0x34,  0x88,  0xC9,  0xFD,  0x10,  0xC2,
     0xD7,  0xBC,  0x0B,  0xC0,  0x06,  0xFE,  0xC8,  0x10,
     0x80,  0x84,  0x1A,  0x17,  0x70,  0x5F,  0xF3,  0xCA,
     0x6F,  0xA6,  0x02,  0xA2,  0x51,  0x66,  0xE6,  0xF0,
     0xCA,  0xD2,  0xCC,  0xB6,  0x78,  0xA8,  0xA7,  0xB4,
     0x84,  0x50,  0x4C,  0x98,  0xAA,  0x65,  0x3F,  0x60,
     0x74,  0xF4,  0xCC,  0x51,  0xC8,  0xAB,  0x8F,  0xAB,
     0x93,  0x76,  0x04,  0xBC,  0x8D,  0x38,  0x53,  0x31,
     0xDF,  0x20,  0x44,  0x29,  0xFA,  0xA5,  0xA1,  0x33,
     0x39,  0x7D,  0xF5,  0x14,  0x62,  0x2E,  0x22,  0x06,
     0x98,  0x14,  0x02,  0xE4,  0x40,  0x01,  0x7C,  0x8B,
     0xD3,  0xD5,  0x84,  0x31,  0xF5,  0x90,  0xFB,  0x5C,
     0x39,  0x1B,  0x04,  0xD9,  0xD6,  0xB5,  0x27,  0xCC,
     0x44,  0x63,  0xFE,  0x00,  0xEE,  0xE9,  0x48,  0x6F,
     0xC4,  0x24,  0xED,  0x23,  0x37,  0xE7,  0xED,  0x86,
     0xCF,  0x0D,  0xFF,  0xA3,  0x1F,  0x6A,  0x72,  0xC9
};



/* ======================================================================== */
/*  Variables to catch return values from function.                         */
/* ======================================================================== */
unsigned ret_val_c = 0;

int main(int argc, char** argv)
{
        #pragma monitor start
        #pragma kernel
	ret_val_c = IMG_sad_16x16_c(src_data, ref_data, pitch);
        #pragma monitor stop

	if (argc > 42 && ! strcmp(argv[0], ""))	printf("%u", ret_val_c);

	unsigned expected_value = 22179;
	if(ret_val_c == expected_value) {
		return 10;
	}
	else {
		return 1;
	}


}
