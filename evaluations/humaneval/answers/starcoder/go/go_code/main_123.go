package main 
 
import (          "sort" 
 )

func GetOddCollatz(n int) []int {
    odds := []int{}
    for n > 1 {
        if n % 2 == 1 {
            odds = append(odds, n)
        }
        if n % 2 == 0 {
            n = n / 2
        } else {
            n = 3*n + 1
    }
    sort.Ints(odds)
    return odds
}