package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestGetOddCollatz(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{1, 5, 7, 11, 13, 17}, GetOddCollatz(14))
    assert.Equal([]int{1, 5}, GetOddCollatz(5))
    assert.Equal([]int{1, 3, 5}, GetOddCollatz(12))
    assert.Equal([]int{1}, GetOddCollatz(1))
}
