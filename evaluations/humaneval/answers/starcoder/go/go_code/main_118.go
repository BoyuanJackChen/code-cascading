package main 
 
import (          "bytes" 
 )

func GetClosestVowel(word string) string {
    vowels := []byte{'a', 'e', 'i', 'o', 'u'}
    for i := len(word) - 1; i >= 0; i-- {
        if bytes.Contains(vowels, []byte(word[i])) {
            return word[i]
        }
    }
    return ""
}
