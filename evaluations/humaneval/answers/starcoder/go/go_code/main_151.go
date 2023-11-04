package main 
 
import (          "math" 
 )

func DoubleTheDifference(lst []float64) int {
    var sum int
    for _, v := range lst {
        if v < 0 || math.Mod(v, 1)!= 0 {
            continue
        }
        sum += v * v
    }
    return sum
}