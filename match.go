package main

import (
	"image"
	"log"
	"path/filepath"

	"github.com/pkg/errors"
	"gocv.io/x/gocv"
)

const (
	templateGlob = "templates/*.png"
	exampleGlob  = "examples/*"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	templates, err := filepath.Glob(templateGlob)
	if err != nil {
		return err
	}

	tmpls := map[string]gocv.Mat{}
	for _, tmpl := range templates {
		im := gocv.IMRead(tmpl, gocv.IMReadGrayScale)
		if im.Empty() {
			return errors.Errorf("failed to load %q", tmpl)
		}
		tmpls[filepath.Base(tmpl)] = im
	}

	examples, err := filepath.Glob(exampleGlob)
	if err != nil {
		return err
	}
	for _, ex := range examples {
		im := gocv.IMRead(ex, gocv.IMReadGrayScale)
		if im.Empty() {
			return errors.Errorf("failed to load %q", ex)
		}
		gocv.Resize(im, &im, image.Point{1920, 1080}, 0, 0, gocv.InterpolationDefault)

		for tmplName, tmpl := range tmpls {
			result := gocv.NewMat()
			gocv.MatchTemplate(im, tmpl, &result, gocv.TmCcoeffNormed, gocv.NewMat())
			_, maxConfidence, _, _ := gocv.MinMaxLoc(result)
			if maxConfidence > 0.85 {
				log.Printf("%s: %s - confidence %f", filepath.Base(ex), tmplName, maxConfidence)
			}
		}
	}

	return nil
}
