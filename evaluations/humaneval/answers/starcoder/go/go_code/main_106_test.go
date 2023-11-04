package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestF(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{1, 2, 6, 24, 15}, F(5))
    assert.Equal([]int{1, 2, 6, 24, 15, 720, 28}, F(7))
    assert.Equal([]int{1}, F(1))
    assert.Equal([]int{1,2,6}, F(3))
}
