package main

import (
	"fmt"
	"image"
	"image/png"
	"os"
)

func ingestPng(fp string) []float64 {
	file, err := os.Open(fp)
	defer file.Close()
	if err != nil {
		fmt.Println("Error reading png", err)
	}
	img, err := png.Decode(file)
	if err != nil {
		fmt.Println("Error reading png 2", err)
	}
	bounds := img.Bounds()
	grayScale := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			pixel := img.At(x, y)
			grayScale.Set(x, y, pixel)
		}
	}
	pixelArray := make([]float64, len(grayScale.Pix))

	for i, p := range grayScale.Pix {
		pixelArray[i] = (float64(255-p) / 255.0 * 0.99) + 0.01
	}
	return pixelArray
}
