package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

const (
	chunkSize    = 500
	maxChunks    = 100
	ollamaUrl    = "http://localhost:11434/api/generate"
	embedderUrl  = "http://localhost:11434/api/embeddings"
	modelName    = "mistral:7b-instruct-q2_K"
	embedderName = "nomic-embed-text:latest"
)

type EmbeddingChunk struct {
	Text      string
	Embedding []float32
}

var chunks []EmbeddingChunk

func extractTextFromPDF(path string) (string, error) {
	cmd := exec.Command("pdftotext", "-q", path, "-")
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("pdftotext error: %v", err)
	}
	return cleanText(out.String()), nil
}

func cleanText(text string) string {
	text = regexp.MustCompile(`[\x{000d}\x{000c}\x{000a}]+`).ReplaceAllString(text, " ")
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func splitText(text string) []string {
	words := strings.Fields(text)
	var chunks []string
	currentChunk := strings.Builder{}
	
	for _, word := range words {
		if currentChunk.Len()+len(word) > chunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			if len(chunks) >= maxChunks {
				break
			}
		}
		currentChunk.WriteString(word + " ")
	}
	
	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}
	
	return chunks
}

func generateEmbeddings(text string) ([]float32, error) {
	requestBody := map[string]interface{}{
		"model":  embedderName,
		"prompt": text,
	}

	jsonData, _ := json.Marshal(requestBody)
	resp, err := http.Post(embedderUrl, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Embedding, nil
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, magA, magB float32
	for i := range a {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	return dot / (float32(math.Sqrt(float64(magA))) * float32(math.Sqrt(float64(magB))))
}

func findRelevantChunks(query string, topK int) []string {
	queryEmbedding, err := generateEmbeddings(query)
	if err != nil {
		return nil
	}

	type scoredChunk struct {
		text  string
		score float32
	}

	var scoredChunks []scoredChunk
	for _, chunk := range chunks {
		score := cosineSimilarity(queryEmbedding, chunk.Embedding)
		scoredChunks = append(scoredChunks, scoredChunk{
			text:  chunk.Text,
			score: score,
		})
	}

	// Sort by score descending
	for i := 0; i < len(scoredChunks); i++ {
		for j := i + 1; j < len(scoredChunks); j++ {
			if scoredChunks[i].score < scoredChunks[j].score {
				scoredChunks[i], scoredChunks[j] = scoredChunks[j], scoredChunks[i]
			}
		}
	}

	var bestChunks []string
	for i := 0; i < topK && i < len(scoredChunks); i++ {
		bestChunks = append(bestChunks, scoredChunks[i].text)
	}

	return bestChunks
}

func main() {
	text, err := extractTextFromPDF("Understanding_Climate_Change.pdf")
	if err != nil {
		log.Fatal(err)
	}

	textChunks := splitText(text)
	
	for _, chunk := range textChunks {
		embedding, err := generateEmbeddings(chunk)
		if err != nil {
			log.Printf("Embedding generation error: %v", err)
			continue
		}
		chunks = append(chunks, EmbeddingChunk{
			Text:      chunk,
			Embedding: embedding,
		})
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("مرحبًا! اطرح أسئلة عن الكتاب (اكتب 'خروج' للإنهاء):")

	for {
		fmt.Print("\nأنت: ")
		scanner.Scan()
		query := scanner.Text()

		if strings.ToLower(query) == "خروج" {
			break
		}

		contextChunks := findRelevantChunks(query, 3)
		context := strings.Join(contextChunks, "\n")

		prompt := fmt.Sprintf(`استخدم السياق التالي للإجابة على السؤال:
%s

السؤال: %s
الإجابة:`, context, query)

		requestBody := map[string]interface{}{
			"model":  modelName,
			"prompt": prompt,
			"stream": false,
		}

		jsonData, _ := json.Marshal(requestBody)
		resp, err := http.Post(ollamaUrl, "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			log.Printf("API error: %v", err)
			continue
		}
		defer resp.Body.Close()

		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			log.Printf("Response error: %v", err)
			continue
		}

		fmt.Printf("\nالبوت: %s\n", result["response"])
	}
}