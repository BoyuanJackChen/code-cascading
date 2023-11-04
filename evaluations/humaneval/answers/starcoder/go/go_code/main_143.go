package main 
 
import (          "strings" 
 )

func WordsInSentence(sentence string) string {
    words := strings.Split(sentence, " ")
    primeWords := []string{}
    for _, word := range words {
        if isPrime(len(word)) {
            primeWords = append(primeWords, word)
        }
    }
    return strings.Join(primeWords, " ")
}