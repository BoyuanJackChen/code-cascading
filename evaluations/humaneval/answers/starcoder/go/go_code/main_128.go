package main 
 
import (          "math" 
 )

func ProdSigns(arr []int) interface{} {
    if len(arr) == 0 {
        return nil
    }
    prod := 1
    sum := 0
    for _, v := range arr {
        if v == 0 {
            prod = 0
        } else {
            prod *= v
            sum += math.Abs(float64(v))
        }
    }
    return sum * prod
}
