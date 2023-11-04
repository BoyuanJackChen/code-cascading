package main 
 
import (          "strconv" 
 )

func EvenOddPalindrome(n int) [2]int {
    even, odd := 0, 0
    for i := 1; i <= n; i++ {
        s := strconv.Itoa(i)
        if s == Reverse(s) {
            if i % 2 == 0 {
                even++
            } else {
                odd++
            }
        }
    }
    return [2]int{even, odd}
}