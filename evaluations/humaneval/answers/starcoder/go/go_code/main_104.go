package main 
 
import (          "sort" 
 )

func UniqueDigits(x []int) []int {
    var result []int
    for _, v := range x {
        if!hasEvenDigit(v) {
            result = append(result, v)
        }
    }
    sort.Ints(result)
    return result
}