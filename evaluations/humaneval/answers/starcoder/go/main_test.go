package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestGenerateIntegers(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{2, 4, 6, 8}, GenerateIntegers(2, 10))
    assert.Equal([]int{2, 4, 6, 8}, GenerateIntegers(10, 2))
    assert.Equal([]int{2, 4, 6, 8}, GenerateIntegers(132, 2))
    assert.Equal([]int{}, GenerateIntegers(17, 89))
}
