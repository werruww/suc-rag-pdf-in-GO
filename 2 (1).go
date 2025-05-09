package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

const (
	ollamaUrl = "http://localhost:11434/api/generate"
	modelName = "mistral:7b-instruct-q2_K" // أو أي نموذج مدعوم
	bookFile  = "1.txt" // ملف النص الخاص بالكتاب
)

// تحميل محتوى الكتاب من ملف
func loadBookContent() (string, error) {
	content, err := os.ReadFile(bookFile)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

// توليد الردود باستخدام Ollama
func generateResponse(question, context string) (string, error) {
	prompt := fmt.Sprintf(`استخدم المعلومات التالية من الكتاب للإجابة على السؤال:
%s

السؤال: %s
الإجابة:`, context, question)

	requestBody := map[string]interface{}{
		"model":  modelName,
		"prompt": prompt,
		"stream": false,
	}

	jsonData, _ := json.Marshal(requestBody)
	resp, err := http.Post(ollamaUrl, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	return result["response"].(string), nil
}

// تشغيل واجهة الشات
func runChatSession(bookContent string) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("\nمرحبًا! يمكنك طرح أسئلة عن الكتاب. اكتب 'خروج' للإنهاء.")

	for {
		fmt.Print("\nأنت: ")
		scanner.Scan()
		question := scanner.Text()

		if strings.ToLower(question) == "خروج" {
			break
		}

		response, err := generateResponse(question, bookContent)
		if err != nil {
			log.Printf("خطأ في توليد الرد: %v", err)
			continue
		}

		fmt.Printf("\nالبوت: %s\n", response)
	}
}

func main() {
	// تحميل محتوى الكتاب
	bookContent, err := loadBookContent()
	if err != nil {
		log.Fatalf("خطأ في قراءة الملف: %v", err)
	}

	// بدء جلسة الحوار
	runChatSession(bookContent)
}