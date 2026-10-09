// Command attachments sends files to a model next to text: an embedded CSV
// (FileFS), a chart drawn in memory (Data), files named on the command line
// (File) and, with -serve, files uploaded over HTTP (Reader).
package main

import (
	"bytes"
	"context"
	"embed"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"net/http"
	"os"
	"os/signal"

	"crux.foo"
)

//go:embed data
var data embed.FS

func main() {
	model := flag.String("model", crux.ClaudeHaiku4_5, "Model ID; the provider is inferred from it")
	serve := flag.String("serve", "", "Address to serve an upload form on, such as :8080")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	agent := crux.Must(crux.New("analyst", *model,
		crux.WithInstructions("You are a careful data analyst. Answer in a few short sentences."),
	))

	if *serve != "" {
		serveUploads(ctx, agent, *serve)
		return
	}

	session := crux.MustSession(crux.NewSession(ctx, agent))

	// The chart was drawn from different numbers than the CSV holds, so a
	// model that reads both should notice that Q3 doesn't match.
	answer, err := session.Run(ctx,
		"Here is our quarterly revenue as a CSV and as a bar chart. Do they agree? If not, which quarter differs?",
		crux.FileFS(data, "data/sales.csv"),
		crux.Data(barChart([]int{120, 135, 90, 165})).WithName("revenue.png"),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n\n", answer)

	// Files named on the command line, such as a PDF report or a photo.
	if paths := flag.Args(); len(paths) > 0 {
		inputs := []any{"Summarise each of these files in one sentence."}
		for _, path := range paths {
			inputs = append(inputs, crux.File(path))
		}
		answer, err = session.Run(ctx, inputs...)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(answer)
	}
}

// serveUploads answers questions about uploaded files. Each request gets its
// own session; the upload is read once, when Run starts.
func serveUploads(ctx context.Context, agent *crux.Agent, addr string) {
	http.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `<form method="post" enctype="multipart/form-data">
<p><input name="question" size="60" value="What is in this file?"></p>
<p><input type="file" name="file"> <button>Ask</button></p>
</form>`)
	})
	http.HandleFunc("POST /", func(w http.ResponseWriter, r *http.Request) {
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		defer file.Close()

		session, err := crux.NewSession(r.Context(), agent)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		answer, err := session.Run(r.Context(), r.FormValue("question"), crux.Reader(file).WithName(header.Filename))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		fmt.Fprintln(w, answer)
	})

	server := &http.Server{Addr: addr}
	go func() {
		<-ctx.Done()
		server.Close()
	}()
	log.Printf("upload a file at http://localhost%s", addr)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

// barChart draws one bar per value as a PNG, scaled so 200 fills the height.
func barChart(values []int) []byte {
	const width, height, bar, gap = 400, 240, 60, 30
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	blue := image.NewUniform(color.RGBA{R: 40, G: 100, B: 200, A: 255})
	for i, v := range values {
		x := gap + i*(bar+gap)
		top := height - v*height/200
		draw.Draw(img, image.Rect(x, top, x+bar, height), blue, image.Point{}, draw.Src)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		log.Fatal(err)
	}
	return buf.Bytes()
}
