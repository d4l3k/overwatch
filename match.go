package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"path/filepath"
	"strings"

	vision "cloud.google.com/go/vision/apiv1"
	"github.com/otiai10/gosseract"
	"github.com/pkg/errors"
	"gocv.io/x/gocv"
)

const (
	templateGlob = "templates/*.png"
	exampleGlob  = "examples/*"

	smallFactor    = 0.75
	matchThreshold = 0.85
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ocr := gosseract.NewClient()
	defer ocr.Close()
	ocr.SetWhitelist("0123456789.%QWERTYUIOPASDFGHJKLZXCVBNM|:")
	ocr.SetLanguage("eng", "Futura", "BigNoodleToo", "BigNoodleTooOblique")
	ocr.SetPageSegMode(gosseract.PSM_SPARSE_TEXT)

	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return err
	}
	_ = client

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
		defer im.Close()

		name := strings.Split(filepath.Base(tmpl), ".")[0]
		tmpls[name] = im

		small := gocv.NewMat()
		defer small.Close()
		gocv.Resize(im, &small, image.Point{int(float64(im.Cols()) * smallFactor), int(float64(im.Rows()) * smallFactor)}, 0, 0, gocv.InterpolationDefault)
		tmpls[name+"-sm"] = small
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
		defer im.Close()

		gocv.Resize(im, &im, image.Point{1920, 1080}, 0, 0, gocv.InterpolationDefault)

		out := im.Clone()
		defer out.Close()

		/*
			file, err := os.Open(ex)
			if err != nil {
				return err
			}
			defer file.Close()
			image, err := vision.NewImageFromReader(file)
			if err != nil {
				return err
			}
			annotations, err := client.DetectTexts(ctx, image, nil, 1000)
			if err != nil {
				return err
			}
			log.Printf("%s: Annotations: %+v", filepath.Base(ex), annotations)
		*/

		/*
			largeBytes, err := gocv.IMEncode(gocv.PNGFileExt, im)
			if err != nil {
				return err
			}

			ocr.SetImageFromBytes(largeBytes)
			text, err := ocr.Text()
			if err != nil {
				return err
			}
			log.Printf("%s: Text: \n%s\n---", filepath.Base(ex), text)
		*/

		for tmplName, tmpl := range tmpls {
			result := gocv.NewMat()
			defer result.Close()
			gocv.MatchTemplate(im, tmpl, &result, gocv.TmCcoeffNormed, gocv.NewMat())
			_, maxConfidence, _, maxLoc := gocv.MinMaxLoc(result)
			if maxConfidence > 0.85 {
				log.Printf("%s: %s - confidence %f", filepath.Base(ex), tmplName, maxConfidence)
				gocv.PutText(&out, tmplName, maxLoc, gocv.FontHersheySimplex, 1, color.RGBA{255, 255, 255, 255}, 2)
			}
		}

		bytes, err := gocv.IMEncode(gocv.PNGFileExt, out)
		if err != nil {
			return err
		}
		path := fmt.Sprintf("/tmp/%s", filepath.Base(ex))
		log.Printf("writing to %s", path)
		if err := ioutil.WriteFile(path, bytes, 0755); err != nil {
			return err
		}
	}

	return nil
}
