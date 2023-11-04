package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestUniqueDigits(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{1, 15, 33}, UniqueDigits([]int{15, 33, 1422, 1}))
    assert.Equal([]int{}, UniqueDigits([]int{152, 323, 1422, 10}))
    assert.Equal([]int{111, 151}, UniqueDigits([]int{12345, 2033, 111, 151}))
    assert.Equal([]int{31, 135}, UniqueDigits([]int{135, 103, 31}))
}
