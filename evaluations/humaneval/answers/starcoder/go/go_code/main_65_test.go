package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestCircularShift(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("001", CircularShift(100, 2))
    assert.Equal("12", CircularShift(12, 2))
    assert.Equal("79", CircularShift(97, 8))
    assert.Equal("21", CircularShift(12, 1))
    assert.Equal("11", CircularShift(11, 101))
}
