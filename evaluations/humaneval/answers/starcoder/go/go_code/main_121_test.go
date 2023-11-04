package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSolution(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(12, Solution([]int{5, 8, 7, 1}))
    assert.Equal(9, Solution([]int{3, 3, 3, 3, 3}))
    assert.Equal(0, Solution([]int{30, 13, 24, 321}))
    assert.Equal(5, Solution([]int{5, 9}))
    assert.Equal(0, Solution([]int{2, 4, 8}))
    assert.Equal(23, Solution([]int{30, 13, 23, 32}))
    assert.Equal(3, Solution([]int{3, 13, 2, 9}))
}
