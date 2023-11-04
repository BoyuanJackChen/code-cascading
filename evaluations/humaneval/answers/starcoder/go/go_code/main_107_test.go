package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestEvenOddPalindrome(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([2]int{8,13}, EvenOddPalindrome(123))
    assert.Equal([2]int{4,6}, EvenOddPalindrome(12))
    assert.Equal([2]int{1,2}, EvenOddPalindrome(3))
    assert.Equal([2]int{6,8}, EvenOddPalindrome(63))
    assert.Equal([2]int{5,6}, EvenOddPalindrome(25))
    assert.Equal([2]int{4,6}, EvenOddPalindrome(19))
    assert.Equal([2]int{4,5}, EvenOddPalindrome(9))
    assert.Equal([2]int{0,1}, EvenOddPalindrome(1))
}
