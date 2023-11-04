package main 
 
import (          "strings" 
 )

func ReverseDelete(s,c string) [2]interface{} {
    var result string
    for _, char := range s {
        if!strings.Contains(c, string(char)) {
            result += string(char)
        }
    }
    return [2]interface{}{result, strings.EqualFold(result, Reverse(result))}
}