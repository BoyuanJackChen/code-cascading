package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestRightAngleTriangle(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, RightAngleTriangle(3, 4, 5))
    assert.Equal(false, RightAngleTriangle(1, 2, 3))
    assert.Equal(true, RightAngleTriangle(10, 6, 8))
    assert.Equal(false, RightAngleTriangle(2, 2, 2))
    assert.Equal(true, RightAngleTriangle(7, 24, 25))
    assert.Equal(false, RightAngleTriangle(10, 5, 7))
    assert.Equal(true, RightAngleTriangle(5, 12, 13))
    assert.Equal(true, RightAngleTriangle(15, 8, 17))
    assert.Equal(true, RightAngleTriangle(48, 55, 73))
    assert.Equal(false, RightAngleTriangle(1, 1, 1))
    assert.Equal(false, RightAngleTriangle(2, 2, 10))
}
