package main 
 
import (          "strings" 
 )

func SelectWords(s string, n int) []string {
    words := strings.Fields(s)
    var result []string
    for _, word := range words {
        if countConsonants(word) == n {
            result = append(result, word)
        }
    }
    return result
}