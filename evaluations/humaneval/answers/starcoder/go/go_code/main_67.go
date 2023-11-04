package main 
 
import (          "strings" 
          "strconv" 
 )

func FruitDistribution(s string,n int) int {
    var apples,oranges,mango int
    var str string
    str = strings.Split(s," ")
    apples, _ = strconv.Atoi(str[0])
    oranges, _ = strconv.Atoi(str[2])
    mango = n - apples - oranges
    return mango
}