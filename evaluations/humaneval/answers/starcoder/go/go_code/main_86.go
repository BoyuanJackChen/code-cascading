package main 
 
import (          "strings" 
          "sort" 
 )

func AntiShuffle(s string) string {
    words := strings.Split(s, " ")
    for i, word := range words {
        sort.Strings(word)
        words[i] = word
    }
    return strings.Join(words, " ")
}